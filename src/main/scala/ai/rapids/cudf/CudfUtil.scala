/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf

// This hack should be removed after cuDF 21.12 is released.
// More details: https://github.com/rapidsai/cudf/commit/28d9a5569c411f202680d27c3b4e5b8adb5ad882
object CudfUtil {
  def buildDeviceMemoryBuffer(address: Long, lengthInBytes: Long): DeviceMemoryBuffer = {
   new DeviceMemoryBuffer(address, lengthInBytes, Cuda.DEFAULT_STREAM)
  }

  def buildRmmMemoryBuffer(address: Long, lengthInBytes: Long, rmmBufferAddress: Long): DeviceMemoryBuffer = {
    DeviceMemoryBuffer.fromRmm(address, lengthInBytes, rmmBufferAddress)
  }
}
