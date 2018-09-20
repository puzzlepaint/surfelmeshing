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


#include "libvis/qt_thread.h"

#include <functional>
#include <iostream>

#include <QApplication>
#include <QObject>
#include <QTimer>

namespace vis {

struct QtThreadSignalEmitter : public QObject {
 Q_OBJECT
 public:
  void Emit() {
    emit RunFunctionQueueSignal();
  }
  
 signals:
  void RunFunctionQueueSignal();
};

struct QtThreadFunctionQueueRunner : public QObject {
 Q_OBJECT
 public slots:
  void RunFunctionQueueSlot() {
    function<void()> f;
    
    while (true) {
      unique_lock<mutex> lock(function_queue_mutex_);
      if (function_queue_.empty()) {
        lock.unlock();
        function_queue_empty_condition_.notify_all();
        return;
      }
      f = function_queue_[0];
      lock.unlock();
      
      f();
      
      lock.lock();
      // NOTE: It is important to only delete the function from the queue once
      //       it has been executed. Otherwise, WaitForFunctionQueue() might
      //       assume that the queue is completed when the last function has
      //       been removed from the queue but was not executed yet.
      function_queue_.erase(function_queue_.begin());
      lock.unlock();
    }
  }

 public:
  mutex function_queue_mutex_;
  condition_variable function_queue_empty_condition_;
  vector<function<void()>> function_queue_;
};


QtThread::QtThread() {
  startup_done_ = false;
  qapplication_exit_called_ = false;
  quit_done_ = false;
  qt_thread_.reset(new thread(std::bind(&QtThread::Run, this)));
}

QtThread::~QtThread() {
  if (!quit_done_) {
    std::cout << "Error: QtThread::Quit() must be called before it is destructed!" << std::endl;
  }
}

void QtThread::WaitForStartup() {
  unique_lock<mutex> lock(startup_mutex_);
  while (!startup_done_) {
    startup_condition_.wait(lock);
  }
  lock.unlock();
}

void QtThread::RunInQtThread(const function<void()>& f) {
  if (qapplication_exit_called_) {
    return;
  }
  
  if (std::this_thread::get_id() == qt_thread_->get_id()) {
    // RunInQtThread() is called from the QT thread. Call the function directly.
    f();
  } else {
    WaitForStartup();
    
    unique_lock<mutex> lock(function_queue_runner_->function_queue_mutex_);
    function_queue_runner_->function_queue_.emplace_back(f);
    lock.unlock();
    
    // Invoke a queued connection to run the slot in the QApplication thread.
    QtThreadSignalEmitter emitter;
    QObject::connect(
        &emitter, SIGNAL(RunFunctionQueueSignal()),
        function_queue_runner_.get(), SLOT(RunFunctionQueueSlot()),
        Qt::QueuedConnection);
    emitter.Emit();
    emitter.disconnect();
  }
}

void QtThread::RunInQtThreadBlocking(const function<void()>& f) {
  if (qapplication_exit_called_) {
    return;
  }
  
  if (std::this_thread::get_id() == qt_thread_->get_id()) {
    // RunInQtThreadBlocking() is called from the QT thread. Call the function directly.
    f();
  } else {
    RunInQtThread(f);
    WaitForFunctionQueue();
  }
}

void QtThread::Quit() {
  WaitForStartup();
  
  RunInQtThreadBlocking([&](){
    qapplication_exit_called_ = true;
    QApplication::exit();
  });
  
  unique_lock<mutex> quit_lock(quit_mutex_);
  while (!quit_done_) {
    quit_condition_.wait(quit_lock);
  }
  quit_lock.unlock();
  
  qt_thread_->join();
}

void QtThread::Run() {
  int argc = 0;
  QApplication qapp(argc, nullptr);
  qapp.setQuitOnLastWindowClosed(false);
  
  function_queue_runner_.reset(new QtThreadFunctionQueueRunner());
  
  unique_lock<mutex> startup_lock(startup_mutex_);
  startup_done_ = true;
  startup_lock.unlock();
  startup_condition_.notify_all();
  
  qapp.exec();
  
  unique_lock<mutex> quit_lock(quit_mutex_);
  quit_done_ = true;
  quit_lock.unlock();
  quit_condition_.notify_all();
}

void QtThread::WaitForFunctionQueue() {
  unique_lock<mutex> lock(function_queue_runner_->function_queue_mutex_);
  while (!function_queue_runner_->function_queue_.empty()) {
    function_queue_runner_->function_queue_empty_condition_.wait(lock);
  }
  lock.unlock();
}

QtThread* QtThread::Instance() {
  static QtThread instance;
  return &instance;
}

}

#include "qt_thread.moc"
