/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include <ds3231.h>
#include "esp-idf-ds3231.h"
//#include "i2cdev.h" // ✅ ต้องมี ถ้าจะใช้ i2c_dev_t
// #include "driver/i2c.h"           // ✅ สำหรับ I2C config: i2c_config_t, I2C_NUM_0
#include "driver/i2c.h"
#include "driver/gpio.h" // ✅ สำหรับ GPIO_NUM_xx
#include "esp_log.h"     // ✅ สำหรับ debug เพิ่มเติม (optional)
#include <time.h>
#include <string.h>
// I2C configuration for DS3231
#define I2C_MASTER_NUM I2C_NUM_0
#define I2C_MASTER_SDA_IO GPIO_NUM_16
#define I2C_MASTER_SCL_IO GPIO_NUM_17
#define I2C_MASTER_FREQ_HZ 100000
namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *model_input = nullptr;
  FeatureProvider *feature_provider = nullptr;
  RecognizeCommands *recognizer = nullptr;
  int32_t previous_time = 0;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 100 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  uint8_t feature_buffer[kFeatureElementCount];
  uint8_t *model_input_buffer = nullptr;
} // namespace

// The name of this function is important for Arduino compatibility.
void setup()
{
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.

  static tflite::MicroMutableOpResolver<11> micro_op_resolver;
  if (micro_op_resolver.AddConv2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddShape() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddStridedSlice() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddPack() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddResizeBilinear() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddQuantize() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddDequantize() != kTfLiteOk)
  {
    return;
  }
  if (micro_op_resolver.AddLogistic() != kTfLiteOk)
  {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteUInt8))
  {
    MicroPrintf("Bad input tensor parameters in model");
    if (model_input->dims->size != 2)
    {
      MicroPrintf("input dim size : %d", model_input->dims->size);
    }
    if (model_input->dims->data[1] !=
        (kFeatureSliceCount * kFeatureSliceSize))
    {
      MicroPrintf("input spectogram dim : %d", model_input->dims->data[1]);
    }
    if (model_input->type != kTfLiteUInt8)
    {
      MicroPrintf("input type : %d", model_input->type);
    }
    return;
  }
  model_input_buffer = model_input->data.uint8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;
  /*anothor inite*/
  i2c_master_bus_handle_t *bus_handle =
      (i2c_master_bus_handle_t *)malloc(sizeof(i2c_master_bus_handle_t));
  // Create the i2c_master_bus_config_t struct and assign values.
  i2c_master_bus_config_t i2c_mst_config = {
      .clk_source = I2C_CLK_SRC_DEFAULT,
      .i2c_port = -1,
      .scl_io_num = I2C_MASTER_SCL_IO,
      .sda_io_num = I2C_MASTER_SDA_IO,
      .glitch_ignore_cnt = 7,

      // The DS3231 **requires** pullup resistors on all of its I/O pins.
      // Note: Some DS3231 boards have pullup resistors as part
      // of their design.
      .flags.enable_internal_pullup = true,
  };
  i2c_new_master_bus(&i2c_mst_config, bus_handle);
  rtc_handle_t *rtc_handle = ds3231_init(bus_handle);

  /*try to print*/
  time_t now;
  char strftime_buf[64];
  struct tm timeinfo;
  now = ds3231_time_unix_get(rtc_handle);
  localtime_r(&now, &timeinfo);
  strftime(strftime_buf, sizeof(strftime_buf), "%c", &timeinfo);
  printf("The current time from the DS3231 RTC moldue is: %s\n", strftime_buf);

  MicroPrintf("Setup complete, ready to run inference.");
}

// The name of this function is important for Arduino compatibility.
void loop()
{
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk)
  {
    MicroPrintf("Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0)
  {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++)
  {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    MicroPrintf("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor *output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char *found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk)
  {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  /*
  MicroPrintf("Command: %s, Score: %d, New Command: %d",
              found_command ? found_command : "None", score, is_new_command);
              */
  bool snore = false;
  bool old_snore = false;
  if (score >= 145)
  {
    snore = true;
  }
  else
  {
    snore = false;
  }
  if (snore != old_snore)
  {
    old_snore = snore;
    if (snore)
    {
      MicroPrintf("Snore detected");
    }
    else
    {
      MicroPrintf("Snore stopped");
    }
  }
  // RespondToCommand(current_time, found_command, score, is_new_command);
}
