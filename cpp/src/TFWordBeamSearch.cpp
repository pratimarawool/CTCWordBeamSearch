#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <algorithm>
#include <string>
#include <cctype>
#include <memory>
#include <exception>
#include <thread>
#include <cstddef>
#include <stdint.h>
#include "MatrixTensor.hpp"
#include "WordBeamSearch.hpp"
#include "LanguageModel.hpp"

using namespace tensorflow;

REGISTER_OP("WordBeamSearch")
.Input("mat: float32") // TxBxC
.Input("seq_len: int32") // Bx1
.Attr("beamWidth: int")
.Attr("lmType: string")
.Attr("lmSmoothing: float")
.Attr("corpus: string")
.Attr("chars: string")
.Attr("wordChars: string")
.Output("result: int32")
.Output("score: float")
.Doc(
"Decodes matrix (mat) using a dictionary and language model created from text corpus (corpus). "\
"The softmax function must be applied in advance to mat. "\
"All characters (chars) must be passed in the same order as they appear in mat, not including the CTC-blank. "\
"The characters (wordChars) which can occur in a word are used to create the dictionary and language model from the corpus. "\
"The LM scoring mode (lmType) must be one of the following four strings (not case-sensitive): 'Words', 'NGrams', 'NGramsForecast', 'NGramsForecastAndSample'. "\
"Pass strings UTF8 encoded if using special characters. "
     ) // shape inference adapted from tensorflow CTCGreedyDecoder (Apache License 2.0)
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle inputs;
      ::tensorflow::shape_inference::ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

      // Get batch size from inputs and sequence_length.
      ::tensorflow::shape_inference::DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));

      c->set_output(0, c->Matrix(c->Dim(inputs,1),c->Dim(inputs,0)));
      c->set_output(1, c->Matrix(batch_size, 1));
      return Status::OK();
    });


// custom TF op
class TFWordBeamSearch : public OpKernel 
{
private:
	std::shared_ptr<LanguageModel> m_lm;
	size_t m_beamWidth = 0;
	size_t m_numChars = 0;
	LanguageModelType m_lmType = LanguageModelType::Words;

public:
	// CTOR
	explicit TFWordBeamSearch(OpKernelConstruction* context) 
	:OpKernel(context) 
	{
		// read beam width
		int64 beamWidth64 = 0;
		OP_REQUIRES_OK(context, context->GetAttr("beamWidth", &beamWidth64));
		m_beamWidth = static_cast<size_t>(beamWidth64);
		
		// read type of language model
		std::string strLmType;
		OP_REQUIRES_OK(context, context->GetAttr("lmType", &strLmType));
		
		// map string to enum (default is Words)
		std::transform(strLmType.begin(), strLmType.end(), strLmType.begin(), tolower);
		if(strLmType == "words")
		{
			m_lmType = LanguageModelType::Words;
		}
		else if(strLmType == "ngrams")
		{
			m_lmType = LanguageModelType::NGrams;
		}
		else if(strLmType == "ngramsforecast")
		{
			m_lmType = LanguageModelType::NGramsForecast;
		}
		else if(strLmType == "ngramsforecastandsample")
		{
			m_lmType = LanguageModelType::NGramsForecastAndSample;
		}
		else
		{
			throw std::invalid_argument("unknown LM type (lmType)");
		}

		// read lm smoothing value	
		float lmSmoothing = 0.0f;
		OP_REQUIRES_OK(context, context->GetAttr("lmSmoothing", &lmSmoothing));

		// read corpus (utf8)
		std::string corpus;
		OP_REQUIRES_OK(context, context->GetAttr("corpus", &corpus));

		// read all chars (utf8)
		std::string chars;
		OP_REQUIRES_OK(context, context->GetAttr("chars", &chars));

		// read all chars which occur inside words
		std::string wordChars;
		OP_REQUIRES_OK(context, context->GetAttr("wordChars", &wordChars));

		// create language model
		m_lm = std::make_shared<LanguageModel>(corpus, chars, wordChars, m_lmType, lmSmoothing);

		// query number of chars (may be different to chars.size()) to check tensor shape
		m_numChars = m_lm->getAllChars().size();

		// check string sizes now and the mat size later in the Compute method
		const size_t numWordChars = m_lm->getWordChars().size();
		if(!(numWordChars > 0 && numWordChars < m_numChars))
		{
			throw std::invalid_argument("wordChars must contain at least one character and at least one character less than chars: 0<len(wordChars)<len(chars)");
		}

	}

	
	// fill result from decoder into output tensor
	template<class U>
	void fillResult(const std::vector<uint32_t>& decoded, U& outputMapped, size_t batchElement, size_t maxT, size_t maxC)
	{
		const size_t blank = maxC - 1;

		// write to output tensor
		for(size_t t = 0; t < maxT; ++t)
		{
			outputMapped(batchElement, t) = t < decoded.size() ? decoded[t] : blank;
		}
	}


#ifdef WBS_PARALLEL
	// split batch into groups of batch elements
	std::vector<std::vector<size_t>> splitWork(size_t numThreads, size_t batchSize)
	{
		std::vector<std::vector<size_t>> res(numThreads);

		size_t div = batchSize / numThreads;
		size_t rem = batchSize % numThreads;
		size_t ctrF = 0;
		size_t ctrB = batchSize - 1;
		for(size_t th = 0; th < numThreads; ++th)
		{
			if(rem > 0)
			{
				res[th].push_back(ctrB);
				ctrB--;
				rem--;
			}

			for(size_t i = 0; i < div; ++i)
			{
				res[th].push_back(ctrF);
				ctrF++;
			}
		}

		return res;
	}
#endif


#ifdef WBS_PARALLEL
	// do the work for a group of batch elements
	template<class U, class V>
	void doWork(const U& inputMapped, V& outputMapped, const std::vector<std::vector<size_t>>& work, size_t threadIdx, size_t maxT, size_t maxC)
	{
		const std::vector<size_t>& batchElements = work[threadIdx];
		for(auto b : batchElements)
		{
			// wrapper around Tensor
			MatrixTensor<decltype(inputMapped)> mat(inputMapped, b, maxT, maxC);

			// apply decoding algorithm to batch element 
			const std::shared_ptr<Beam> bestBeam = wordBeamSearch(mat, m_beamWidth, m_lm, m_lmType);
			const std::vector<uint32_t> decoded = bestBeam->getText();
			// write to output tensor
			fillResult(decoded, outputMapped, b, maxT, maxC);
		}
	}
#endif

	
	// computation in TF graph
	void Compute(OpKernelContext* context) override 
	{
		// input: TxBxC, float32
		const Tensor& inputTensor = context->input(0);
    // input: Bx1, int32
    const Tensor& seqLen = context->input(1);
    
		const auto inputShape = inputTensor.shape();
		const size_t maxT = inputShape.dim_size(0);
		const size_t maxB = inputShape.dim_size(1);
		const size_t maxC = inputShape.dim_size(2);
		
		// check tensor size
		if(maxC != m_numChars + 1)
		{
			throw std::invalid_argument("the number of characters (chars) plus 1  must equal dimension 2 of the tensor (mat)");
		}
    
    if (!TensorShapeUtils::IsVector(seqLen.shape())) {
      throw std::invalid_argument("sequence length is not a vector");
      }

    if (!(maxB == seqLen.dim_size(0))) {
      throw std::invalid_argument("length of sequence length is not batch size");
    }

    auto seqLen_t = seqLen.vec<int32>();

    for (unsigned int b = 0; b < maxB; ++b) {
      if (seqLen_t(b)>= 0 && !(seqLen_t(b) <= maxT)) {
        throw std::invalid_argument("sequence length value out of bounds");
      }
    }


		// input tensor
		const auto inputMapped = inputTensor.tensor<float, 3>();

		// output result: BxT, int32
		Tensor* outputTensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({static_cast<int>(maxB), static_cast<int>(maxT)}), &outputTensor));
		auto outputMapped = outputTensor->tensor<int32, 2>();

		// output score: Bx1, float32
		Tensor* scoreTensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({static_cast<int>(maxB), static_cast<int>(1)}), &scoreTensor));

#ifdef WBS_PARALLEL
		// split work into number of threads
		const size_t numThreads = WBS_THREADS;
		const auto work = splitWork(numThreads, maxB);

		// start threads
		std::vector<std::thread> workers;
		for(size_t th = 0; th < numThreads; ++th)
		{
			workers.push_back(std::thread( [&, th] { doWork(inputMapped, outputMapped, work, th, maxT, maxC); } ));
		}

		// wait until all threads finished their work
		for(auto& w : workers)
		{
			w.join();
		}
#else
    auto score_t = scoreTensor->matrix<float>();
    score_t.setZero();

		// go over all batch elements
		for(size_t b = 0; b < maxB; ++b)
		{
			// wrapper around Tensor
			MatrixTensor<decltype(inputMapped)> mat(inputMapped, b, seqLen_t(b), maxC);

			// apply decoding algorithm to batch element 
			const std::shared_ptr<Beam> bestBeam = wordBeamSearch(mat, m_beamWidth, m_lm, m_lmType);
			const std::vector<uint32_t> decoded = bestBeam->getText();
			//LOG(INFO) << "Best beam " << b << " score: " << bestBeam->getTotalProb();
			// write to output tensor
			fillResult(decoded, outputMapped, b, maxT, maxC);
      score_t(b,0) = bestBeam->getTotalProb();
		}
#endif
	}
};


REGISTER_KERNEL_BUILDER(Name("WordBeamSearch").Device(DEVICE_CPU), TFWordBeamSearch);

