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

#include "libvis/opengl_context.h"

#include <GL/glew.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <glog/logging.h>

namespace vis {

struct OpenGLContextImpl {
  Display* display;
  GLXDrawable drawable;
  GLXContext context;
  bool needs_glew_initialization;
};


int XErrorHandler(Display* dsp, XErrorEvent* error) {
  constexpr int kBufferSize = 512;
  char error_string[kBufferSize];
  XGetErrorText(dsp, error->error_code, error_string, kBufferSize);

  LOG(FATAL) << "X Error:\n" << error_string;
  return 0;
}


OpenGLContext::OpenGLContext() { }

OpenGLContext::OpenGLContext(OpenGLContext&& other) { impl.swap(other.impl); }

OpenGLContext& OpenGLContext::operator=(OpenGLContext&& other) {
  impl.swap(other.impl);
  return *this;
}

OpenGLContext::~OpenGLContext() {
  Deinitialize();
}

bool OpenGLContext::InitializeWindowless(OpenGLContext* sharing_context) {
  CHECK(!impl);
  impl.reset(new OpenGLContextImpl());
  
  GLint attributes[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, None};

  int (*old_error_handler)(Display*, XErrorEvent*) =
      XSetErrorHandler(XErrorHandler);

  Display* display = XOpenDisplay(NULL);
  if (!display) {
    LOG(FATAL) << "Cannot connect to X server.";
  }

  Window root_window = DefaultRootWindow(display);
  XVisualInfo* visual = glXChooseVisual(display, 0, attributes);
  if (!visual) {
    LOG(FATAL) << "No appropriate visual found.";
  }

  GLXContext glx_context =
      glXCreateContext(display, visual, sharing_context ? sharing_context->impl->context : nullptr, GL_TRUE);
  if (!glx_context) {
    LOG(FATAL) << "Cannot create GLX context.";
  }
  XFree(visual);

  impl->display = display;
  impl->drawable = root_window;
  impl->context = glx_context;
  impl->needs_glew_initialization = true;

  XSetErrorHandler(old_error_handler);
  return true;
}

void OpenGLContext::Deinitialize() {
  if (!impl || !impl->context) {
    return;
  }
  
  glXDestroyContext(impl->display, impl->context);
  XCloseDisplay(impl->display);

  impl->drawable = None;
  impl->context = nullptr;
  
  impl.reset();
}

void OpenGLContext::AttachToCurrent() {
  impl.reset(new OpenGLContextImpl());
  
  impl->display = glXGetCurrentDisplay();
  impl->drawable = glXGetCurrentDrawable();
  impl->context = glXGetCurrentContext();
  impl->needs_glew_initialization = false;  // TODO: This is not clear.
}

void OpenGLContext::Detach() {
  impl.reset();
}


OpenGLContext SwitchOpenGLContext(const OpenGLContext& context) {
  int (*old_error_handler)(Display*, XErrorEvent*) =
      XSetErrorHandler(XErrorHandler);
  
  OpenGLContext current_context;
  current_context.AttachToCurrent();
  
  if (!current_context.impl->display) {
    // We need a display, otherwise glXMakeCurrent() will segfault.
    current_context.impl->display = context.impl->display;
  }
  
  if (glXMakeCurrent(context.impl->display, context.impl->drawable,
                     context.impl->context) == GL_FALSE) {
    LOG(FATAL) << "Cannot make GLX context current.";
  }
  
  if (context.impl->needs_glew_initialization) {
    // Initialize GLEW on first switch to a context.
    glewExperimental = GL_TRUE;
    GLenum glew_init_result = glewInit();
    CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
    glGetError();  // Ignore GL_INVALID_ENUM​ error caused by glew
    context.impl->needs_glew_initialization = false;
  }
  
  XSetErrorHandler(old_error_handler);
  return current_context;
}

}
