#pragma once
#include "IMatrix.hpp"
#include "LanguageModel.hpp"
#include <stdint.h>
#include <cstddef>
#include "Beam.hpp"

// apply word beam search decoding on the matrix with given beam width
const std::shared_ptr<Beam> wordBeamSearch(const IMatrix& mat, size_t beamWidth, const std::shared_ptr<LanguageModel>& lm, LanguageModelType lmType);

