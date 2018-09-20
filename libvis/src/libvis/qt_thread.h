// Copyright 2018 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "libvis/libvis.h"

namespace vis {

class QtThreadFunctionQueueRunner;

// Starts a thread in which a QApplication is created. Allows to run code within
// this thread. This is necessary because all interaction with Qt GUI objects
// must happen in the thread in which QApplication was created, and we don't
// want to force the user to do this in the main thread. Note that this won't
// work on MacOS according to a comment on the internet: It seems to require
// this to be done in the main thread.
class QtThread {
 public:
  // Constructor. Starts the Qt thread.
  QtThread();
  
  ~QtThread();
  
  // Waits for the QApplication to be created in the Qt thread.
  void WaitForStartup();
  
  // Runs the function in the Qt thread. Does not wait for it to complete.
  void RunInQtThread(const function<void()>& f);
  
  // Runs the function in the Qt thread. Blocks until it completes.
  void RunInQtThreadBlocking(const function<void()>& f);
  
  // Quit() must be called by the application before exiting if it uses the
  // Qt thread. This is because the destructor already seems to be called too
  // late to do the required cleanup. (Even if atexit() is used after waiting
  // for the thread startup, the QApplication destructor will fail if windows
  // are allocated afterwards. atexit() after allocating the last window works,
  // but of course we do not know here when the user allocates the last window.)
  // TODO: It would be nice to have a solution which does not require this call.
  void Quit();
  
  // Returns the global QtThread instance.
  static QtThread* Instance();
  
 protected:
  // Main function of the thread.
  void Run();

 private:
  // Waits for all functions in the queue to finish executing.
  void WaitForFunctionQueue();
  
  unique_ptr<thread> qt_thread_;
  
  std::atomic<bool> startup_done_;
  mutex startup_mutex_;
  condition_variable startup_condition_;
  
  std::atomic<bool> qapplication_exit_called_;
  std::atomic<bool> quit_done_;
  mutex quit_mutex_;
  condition_variable quit_condition_;
  
  unique_ptr<QtThreadFunctionQueueRunner> function_queue_runner_;
};

}
