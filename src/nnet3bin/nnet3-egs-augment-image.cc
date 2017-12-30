// nnet3bin/nnet3-egs-augment-image.cc

// Copyright      2017  Johns Hopkins University (author:  Daniel Povey)
//                2017  Hossein Hadian
//                2017  Yiwen Shao

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

enum FillMode { kNearest, kReflect };

struct ImageAugmentationConfig {
  int32 num_channels;
  //int32 arch_size;
  BaseFloat horizontal_flip_prob;
  BaseFloat horizontal_shift;
  BaseFloat vertical_shift;
  BaseFloat rotation_degree;
  BaseFloat rotation_prob;
  BaseFloat alpha;
  std::string fill_mode_string;

  ImageAugmentationConfig():
      num_channels(1),
      //arch_size(25000),
      horizontal_flip_prob(0.0),
      horizontal_shift(0.0),
      vertical_shift(0.0),
      rotation_degree(0.0),
      rotation_prob(0.0),
      alpha(0.0),
      fill_mode_string("nearest") { }


  void Register(ParseOptions *po) {
    po->Register("num-channels", &num_channels, "Number of colors in the image."
                 "It is important to specify this (helps interpret the image "
                 "correctly.");
    po->Register("horizontal-flip-prob", &horizontal_flip_prob,
                 "Probability of doing horizontal flip");
    po->Register("horizontal-shift", &horizontal_shift,
                 "Maximum allowed horizontal shift as proportion of image "
                 "width.  Padding is with closest pixel.");
    po->Register("vertical-shift", &vertical_shift,
                 "Maximum allowed vertical shift as proportion of image "
                 "height.  Padding is with closest pixel.");
    po->Register("rotation-degree", &rotation_degree,
                 "Maximum allowed degree to rotate the image");
    po->Register("rotation-prob", &rotation_prob,
                 "Probability of doing rotation");
    po->Register("fill-mode", &fill_mode_string, "Mode for dealing with "
		 "points outside the image boundary when applying transformation. "
		 "Choices = {nearest, reflect}");
    //po->Register("arch-size", &arch_size, "Size of archive.");
    po->Register("alpha", &alpha, "alpha in Beta(alpha, alpha) to get lamda "
                 "for linear interpolation in mixup.");
  }

  void Check() const {
    KALDI_ASSERT(num_channels >= 1);
    KALDI_ASSERT(horizontal_flip_prob >= 0 &&
                 horizontal_flip_prob <= 1);
    KALDI_ASSERT(horizontal_shift >= 0 && horizontal_shift <= 1);
    KALDI_ASSERT(vertical_shift >= 0 && vertical_shift <= 1);
    KALDI_ASSERT(rotation_degree >=0 && rotation_degree <= 180);
    KALDI_ASSERT(rotation_prob >=0 && rotation_prob <= 1);
    KALDI_ASSERT(fill_mode_string == "nearest" || fill_mode_string == "reflect");
  }

  FillMode GetFillMode() const {
    FillMode fill_mode;
    if (fill_mode_string == "reflect") {
      fill_mode = kReflect;
    } else {
      if (fill_mode_string != "nearest") {
	KALDI_ERR << "Choices for --fill-mode are 'nearest' or 'reflect', got: "
		  << fill_mode_string;
      } else {
	fill_mode = kNearest;
      }
    }
    return fill_mode;
  }
};

/**
  This function applies a geometric transformation 'transform' to the image.
  Reference: Digital Image Processing book by Gonzalez and Woods.
  @param [in] transform  The 3x3 geometric transformation matrix to apply.
  @param [in] num_channels  Number of channels (i.e. colors) of the image
  @param [in,out] image  The image matrix to be modified.
                     image->NumRows() is the width (number of x values) in
                     the image; image->NumCols() is the height times number
                     of channels (channel varies the fastest).
 */
void ApplyAffineTransform(MatrixBase<BaseFloat> &transform,
                          int32 num_channels,
                          MatrixBase<BaseFloat> *image,
                          FillMode fill_mode) {
  int32 num_rows = image->NumRows(),
        num_cols = image->NumCols(),
        height = num_cols / num_channels,
        width = num_rows;
  KALDI_ASSERT(num_cols % num_channels == 0);
  Matrix<BaseFloat> original_image(*image);
  for (int32 r = 0; r < width; r++) {
    for (int32 c = 0; c < height; c++) {
      // (r_old, c_old) is the coordinate of the pixel in the original image
      // while (r, c) is the coordinate in the new (transformed) image.
      BaseFloat r_old = transform(0, 0) * r +
                                          transform(0, 1) * c + transform(0, 2);
      BaseFloat c_old = transform(1, 0) * r +
                                          transform(1, 1) * c + transform(1, 2);
      // We are going to do bilinear interpolation between 4 closest points
      // to the point (r_old, c_old) of the original image. We have:
      // r1  <=  r_old  <=  r2
      // c1  <=  c_old  <=  c2
      int32 r1 = static_cast<int32>(floor(r_old));
      int32 c1 = static_cast<int32>(floor(c_old));
      int32 r2 = r1 + 1;
      int32 c2 = c1 + 1;

      // These weights determine how much each of the 4 points contributes
      // to the final interpolated value:
      BaseFloat weight_11 = (r2 - r_old) * (c2 - c_old),
          weight_12 = (r2 - r_old) * (c_old - c1),
          weight_21 = (r_old - r1) * (c2 - c_old),
          weight_22 = (r_old - r1) * (c_old - c1);
      // Handle edge conditions:
      if (fill_mode == kNearest) {
        if (r1 < 0) {
          r1 = 0;
          if (r2 < 0) r2 = 0;
        }
        if (r2 >= width) {
          r2 = width - 1;
          if (r1 >= width) r1 = width - 1;
        }
        if (c1 < 0) {
          c1 = 0;
          if (c2 < 0) c2 = 0;
        }
        if (c2 >= height) {
          c2 = height - 1;
          if (c1 >= height) c1 = height - 1;
        }
      } else {
        KALDI_ASSERT(fill_mode == kReflect);
        if (r1 < 0) {
          r1 = - r1;
          if (r2 < 0) r2 = - r2;
        }
        if (r2 >= width) {
          r2 = 2 * width - 2 - r2;
          if (r1 >= width) r1 = 2 * width - 2 - r1;
        }
        if (c1 < 0) {
          c1 = - c1;
          if (c2 < 0) c2 = -c2;
        }
        if (c2 >= height) {
          c2 = 2 * height - 2 - c2;
          if (c1 >= height) c1 = 2 * height - 2 - c1;
        }
      }
      for (int32 ch = 0; ch < num_channels; ch++) {
        // find the values at the 4 points
        BaseFloat p11 = original_image(r1, num_channels * c1 + ch),
            p12 = original_image(r1, num_channels * c2 + ch),
            p21 = original_image(r2, num_channels * c1 + ch),
            p22 = original_image(r2, num_channels * c2 + ch);
        (*image)(r, num_channels * c + ch) = weight_11 * p11 + weight_12 * p12 +
            weight_21 * p21 + weight_22 * p22;
      }
    }
  }
}

/**
   This function randomly modifies (perturbs) the image by applying different
   geometric transformations according to the options in 'config'.
   References: "Digital Image Processing book by Gonzalez and Woods" and
   "Keras: github.com/fchollet/keras/blob/master/keras/preprocessing/image.py"
   @param [in] config  Configuration class that says how
   to perturb the image.
   @param [in,out] image  The image matrix to be modified.
   image->NumRows() is the width (number of x values) in
   the image; image->NumCols() is the height times number
   of channels/colors (channel varies the fastest).
*/
void PerturbImage(const ImageAugmentationConfig &config,
                  MatrixBase<BaseFloat> *image) {
  config.Check();
  FillMode fill_mode = config.GetFillMode();
  int32 image_width = image->NumRows(),
      num_channels = config.num_channels,
      image_height = image->NumCols() / num_channels;
  if (image->NumCols() % num_channels != 0) {
    KALDI_ERR << "Number of columns in image must divide the number "
        "of channels";
  }
  // We do an affine transform which
  // handles flipping, translation, rotation, magnification, and shear.
  Matrix<BaseFloat> transform_mat(3, 3, kUndefined);
  transform_mat.SetUnit();

  Matrix<BaseFloat> shift_mat(3, 3, kUndefined);
  shift_mat.SetUnit();
  // translation (shift) mat:
  // [ 1   0  x_shift
  //   0   1  y_shift
  //   0   0  1       ]
  BaseFloat horizontal_shift = (2.0 * RandUniform() - 1.0) *
      config.horizontal_shift * image_width;
  BaseFloat vertical_shift = (2.0 * RandUniform() - 1.0) *
      config.vertical_shift * image_height;
  shift_mat(0, 2) = round(horizontal_shift);
  shift_mat(1, 2) = round(vertical_shift);
  // since we will center the image before applying the transform,
  // horizontal flipping is simply achieved by setting [0, 0] to -1:
  if (WithProb(config.horizontal_flip_prob))
    shift_mat(0, 0) = -1.0;

  Matrix<BaseFloat> rotation_mat(3, 3, kUndefined);
  rotation_mat.SetUnit();
  // rotation mat:
  // [ cos(theta)  -sin(theta)  0
  //   sin(theta)  cos(theta)   0
  //   0           0            1 ]
  if (RandUniform() <= config.rotation_prob) {
    BaseFloat theta = (2 * config.rotation_degree * RandUniform() -
		       config.rotation_degree) / 180.0 * M_PI;
    rotation_mat(0, 0) = cos(theta);
    rotation_mat(0, 1) = -sin(theta);
    rotation_mat(1, 0) = sin(theta);
    rotation_mat(1, 1) = cos(theta);
  }

  Matrix<BaseFloat> shear_mat(3, 3, kUndefined);
  shear_mat.SetUnit();
  // shear mat:
  // [ 1    -sin(shear)   0
  //   0     cos(shear)   0
  //   0     0            1 ]

  Matrix<BaseFloat> zoom_mat(3, 3, kUndefined);
  zoom_mat.SetUnit();
  // zoom mat:
  // [ x_zoom   0   0
  //   0   y_zoom   0
  //   0     0      1 ]

  // transform_mat = rotation_mat * shift_mat * shear_mat * zoom_mat:
  transform_mat.AddMatMat(1.0, shift_mat, kNoTrans,
                          shear_mat, kNoTrans, 0.0);
  transform_mat.AddMatMatMat(1.0, rotation_mat, kNoTrans,
                             transform_mat, kNoTrans,
                             zoom_mat, kNoTrans, 0.0);
  if (transform_mat.IsUnit())  // nothing to do
    return;

  // we should now change the origin of transform to the center of
  // the image (necessary for flipping, zoom, shear, and rotation)
  // we do this by using two translations: one before the main transform
  // and one after.
  Matrix<BaseFloat> set_origin_mat(3, 3, kUndefined);
  set_origin_mat.SetUnit();
  set_origin_mat(0, 2) = image_width / 2.0 - 0.5;
  set_origin_mat(1, 2) = image_height / 2.0 - 0.5;
  Matrix<BaseFloat> reset_origin_mat(3, 3, kUndefined);
  reset_origin_mat.SetUnit();
  reset_origin_mat(0, 2) = -image_width / 2.0 + 0.5;
  reset_origin_mat(1, 2) = -image_height / 2.0 + 0.5;

  // transform_mat = set_origin_mat * transform_mat * reset_origin_mat
  transform_mat.AddMatMatMat(1.0, set_origin_mat, kNoTrans,
                             transform_mat, kNoTrans,
                             reset_origin_mat, kNoTrans, 0.0);
  ApplyAffineTransform(transform_mat, config.num_channels, image, fill_mode);
}


/**
   This function does image perturbation as directed by 'config'
   The example 'eg' is expected to contain a NnetIo member with the
   name 'input', representing an image.
*/
void PerturbImageInNnetExample(
    const ImageAugmentationConfig &config,
    NnetExample *eg) {
  int32 io_size = eg->io.size();
  bool found_input = false;
  for (int32 i = 0; i < io_size; i++) {
    NnetIo &io = eg->io[i];
    if (io.name == "input") {
      found_input = true;
      Matrix<BaseFloat> image;
      io.features.GetMatrix(&image);
      // note: 'GetMatrix' may uncompress if it was compressed.
      // We won't recompress, but this won't matter because this
      // program is intended to be used as part of a pipe, we
      // likely won't be dumping the perturbed data to disk.
      PerturbImage(config, &image);

      // modify the 'io' object.
      io.features = image;
    }
  }
  if (!found_input)
    KALDI_ERR << "Nnet example to perturb had no NnetIo object named 'input'";
}

float RandBeta(float alpha) {
  std::default_random_engine generator;
  generator.seed(RandInt(0, RAND_MAX));
  std::gamma_distribution<double> distribution(alpha, 1.0);
  float x = distribution(generator);
  float number = (x / (x + distribution(generator)));
  return number;
}
  
void MixupImage(MatrixBase<BaseFloat> *image1,
		MatrixBase<BaseFloat> *image2,
		BaseFloat lamda) {
  
  int32 num_rows = image1->NumRows(),
        num_cols = image1->NumCols(),
        height = num_cols,
        width = num_rows;
  Matrix<BaseFloat> original_image(*image1);
  for (int32 r = 0; r < width; r++) {
    for (int32 c = 0; c < height; c++) {
      (*image1)(r, c) = lamda * original_image(r, c) +
	(1-lamda) * (*image2)(r, c);
    }
  }
}

void MixupImageInNnetExample(
    const ImageAugmentationConfig &config,
    NnetExample *eg1,
    NnetExample *eg2) {
  BaseFloat lamda = RandBeta(config.alpha);
  KALDI_LOG << lamda;
  int32 io_size1 = eg1->io.size(),
    io_size2 = eg1->io.size();
  KALDI_ASSERT(io_size1 == io_size2);
  //bool found_input1 = false;
  //bool found_input2 = false;
  for (int32 i = 0; i < io_size1; i++) {
    NnetIo &io1 = eg1->io[i];
    if (io1.name == "input") {
      Matrix<BaseFloat> image1;
      io1.features.GetMatrix(&image1);
      for (int32 j = 0; j < io_size2; j++) {
	NnetIo &io2 = eg2->io[j];
	if (io2.name == "input") {
	  Matrix<BaseFloat> image2;
	  io2.features.GetMatrix(&image2);
	  // new mixup input image
	  MixupImage(&image1, &image2, lamda);
	  io1.features = image1;
	}
      }
    } else if (io1.name == "output") {
      SparseMatrix<BaseFloat> old_output1(io1.features.GetSparseMatrix());
      const SparseVector<BaseFloat> &row1(old_output1.Row(0));
      int32 num_classes = io1.features.NumCols();
      int32 old_class1;
      BaseFloat value1 = row1.Max(&old_class1); // 1.0
      for (int32 j = 0; j < io_size2; j++) {
	NnetIo &io2 = eg2->io[j];
	if (io2.name == "output") {
	  SparseMatrix<BaseFloat> old_output2(io2.features.GetSparseMatrix());
	  const SparseVector<BaseFloat> &row2(old_output2.Row(0));
	  int32 old_class2;
	  BaseFloat value2 = row2.Max(&old_class2); //1.0
	  // new mixup output
	  BaseFloat new_value1 = lamda * value1,
	    new_value2 = (1-lamda) * value2;
	  std::vector<std::vector<std::pair<MatrixIndexT, BaseFloat> > > new_pairs(1);
	  new_pairs[0].push_back(std::pair<MatrixIndexT, BaseFloat>(old_class1, new_value1));
	  new_pairs[0].push_back(std::pair<MatrixIndexT, BaseFloat>(old_class2, new_value2));
       	  SparseMatrix<BaseFloat> new_output(num_classes, new_pairs);
	  // for (int32 k=0; k < new_pairs[0].size(); k++) {
	  //   KALDI_LOG << new_pairs[0][k].first << " " << new_pairs[0][k].second;
	  // }
	  io2.features.SwapSparseMatrix(&new_output);
	}
      }
    }
  }  
}

// std::string GetRandomKey(const ImageAugmentationConfig &config) {
//   int32 range = config.arch_size;
//   int32 number = RandInt(1, range);
//   std::string key = std::to_string(number);
//   int32 key_length = key.length(),
//     full_length = std::to_string(range).length();
//   for (int i = 0; i < full_length - key_length; i++) {
//     key = '0' + key;
//   }
//   return key;
// }
  

// std::vector<std::string> GetKeyList(SequentialNnetExampleReader *example_reader) {
//   std::vector<std::string> keylist;
//   for (; !example_reader->Done(); example_reader->Next()) {
//     std::string key = example_reader->Key();
//     keylist.push_back(key);
//     KALDI_LOG << key;
//   }
//   return keylist;
// }

// std::string GetRandomKey(std::vector<std::string> keylist) {
//   int64 list_size = keylist.size();
//   int64 random_index = RandInt(0, list_size-1);
//   return keylist[random_index];
// }

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (single frames or fixed-size groups of frames) for neural\n"
        "network training, doing image augmentation inline (copies after possibly\n"
        "modifying of each image, randomly chosen according to configuration\n"
        "parameters).\n"
        "E.g.:\n"
        "  nnet3-egs-augment-image --horizontal-flip-prob=0.5 --horizontal-shift=0.1\\\n"
        "       --vertical-shift=0.1 --srand=103 --num-channels=3 --fill-mode=nearest ark:- ark:-\n"
        "\n"
        "Requires that each eg contain a NnetIo object 'input', with successive\n"
        "'t' values representing different x offsets , and the feature dimension\n"
        "representing the y offset and the channel (color), with the channel\n"
        "varying the fastest.\n"
        "See also: nnet3-copy-egs\n";


    int32 srand_seed = 0;

    ImageAugmentationConfig config;

    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for the random number generator");

    config.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
      //example_reader_tmp(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    //RandomAccessNnetExampleReader example_reader2(examples_rspecifier);
    //std::vector<std::string> keylist = GetKeyList(&example_reader_tmp);

    // for (int32 i=1; i<=300; i++) {
    //   std::string key = std::to_string(i);
    //   int32 key_length = key.length();
    //   for (int32 z=0; z<(5-key_length); z++) {
    // 	key = '0' + key;
    //   }
    //   if (example_reader2.HasKey(key)) {
    // 	KALDI_LOG << "has key "<< key;
    //   } else {
    // 	KALDI_LOG << "no such key " << key;
 
    
    int64 num_done = 0;
    for (; !example_reader.Done(); num_done++) {
      std::string key = example_reader.Key();
      NnetExample eg(example_reader.Value());
      example_reader.Next();
      NnetExample eg2(example_reader.Value());
      std::string key2 = example_reader.Key();
      MixupImageInNnetExample(config, &eg, &eg2);
      example_writer.Write(key, eg);
      KALDI_LOG << "Mixup " << key << " with " << key2;
      //PerturbImageInNnetExample(config, &eg);
      //example_writer.Write(key, eg);
    }
    KALDI_LOG << "Perturbed " << num_done << " neural-network training images.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
