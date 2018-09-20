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


#include "libvis/image.h"

#include "libvis/image_display.h"
#include "libvis/image_io.h"

namespace vis {

ImageIOLibPngRegistrator image_io_libpng_registrator_;
ImageIONetPBMRegistrator image_io_netpbm_registrator_;
#ifdef LIBVIS_HAVE_QT
ImageIOQtRegistrator image_io_qt_registrator_;
#endif

template<>
bool Image<u8>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<u16>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<Vec3u8>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<Vec4u8>::Write(const string& image_file_name) const {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Write(image_file_name, *this);
}

template<>
bool Image<u8>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

template<>
bool Image<u16>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

template<>
bool Image<Vec3u8>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

template<>
bool Image<Vec4u8>::Read(const string& image_file_name) {
  ImageIO* io = ImageIORegistry::Instance()->Get(image_file_name);
  if (!io) {
    LOG(ERROR) << "No image I/O handler exists.";
    return false;
  }
  return io->Read(image_file_name, this);
}

#ifdef LIBVIS_HAVE_QT
template<>
QImage Image<u8>::WrapInQImage() const {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
  return WrapInQImage(QImage::Format_Grayscale8);
#else
  // QImage::Format_Grayscale8 not supported. Create a grayscale color table
  // mapping index i to color (i, i, i) to simulate it.
  // NOTE: It would be preferable to make the color table static.
  QVector<QRgb> colors(256);
  for (int i = 0; i < 256; ++ i) {
    colors[i] = qRgba(i, i, i, 255);
  }
  QImage image(reinterpret_cast<const u8*>(data()), width(), height(),
               stride(), QImage::Format_Indexed8);
  image.setColorTable(colors);
  return image;
#endif
}

template<>
QImage Image<Vec3u8>::WrapInQImage() const {
  return WrapInQImage(QImage::Format_RGB888);
}

template<>
QImage Image<Vec4u8>::WrapInQImage() const {
  return WrapInQImage(QImage::Format_RGBA8888);
}
#endif

template<>
shared_ptr<ImageDisplay> Image<u8>::DebugDisplay(const string& title) const {
#ifdef LIBVIS_HAVE_QT
  shared_ptr<ImageDisplay> display(new ImageDisplay());
  display->Update(*this, title);
  return display;
#else
  (void) title;
  LOG(ERROR) << "Not compiled with debug display support.";
  return shared_ptr<ImageDisplay>();
#endif
}

template<>
shared_ptr<ImageDisplay> Image<Vec3u8>::DebugDisplay(const string& title) const {
#ifdef LIBVIS_HAVE_QT
  shared_ptr<ImageDisplay> display(new ImageDisplay());
  display->Update(*this, title);
  return display;
#else
  (void) title;
  LOG(ERROR) << "Not compiled with debug display support.";
  return shared_ptr<ImageDisplay>();
#endif
}

}
