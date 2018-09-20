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

#include <QMainWindow>

#include "libvis/image_display_qt_widget.h"
#include "libvis/libvis.h"

namespace vis {

class ImageDisplay;

// Qt main window for (debug) image display.
class ImageDisplayQtWindow : public QMainWindow {
 Q_OBJECT
 public:
  ImageDisplayQtWindow(
      ImageDisplay* display,
      QWidget* parent = nullptr,
      Qt::WindowFlags flags = Qt::WindowFlags());
  
  void SetImage(const QImage& image);
  void SetCallbacks(const shared_ptr<ImageWindowCallbacks>& callbacks);
  void SetDisplay(ImageDisplay* display);
  
  void AddSubpixelDotPixelCornerConv(float x, float y, u8 r, u8 g, u8 b);
  void AddSubpixelLinePixelCornerConv(float x0, float y0, float x1, float y1, u8 r, u8 g, u8 b);
  void Clear();
  
  inline const ImageDisplayQtWidget& widget() const { return *image_widget_; }
  inline ImageDisplayQtWidget& widget() { return *image_widget_; }
  
 public slots:
  void CursorPositionChanged(const QPointF& pixel_pos, bool pixel_value_valid, QRgb pixel_value);
  void ZoomChanged(double zoom_factor);
  
  void SaveImage();
  void CopyImage();
  void ResizeToContent(bool adjust_zoom);
  
  inline void ZoomAndResizeToContent() {
    ResizeToContent(true);
  }
  
  inline void ResizeToContentWithoutZoom() {
    ResizeToContent(false);
  }
  
 protected:
  virtual void closeEvent(QCloseEvent* event) override;

 private:
  void UpdateStatusBar();
  
  double last_zoom_factor_;
  bool have_last_cursor_pos_;
  QPointF last_cursor_pos_;
  bool last_pixel_value_valid_;
  QRgb last_pixel_value_;
  
  QToolBar* tool_bar_;
  
  ImageDisplayQtWidget* image_widget_;
  
  ImageDisplay* display_;
};

}
