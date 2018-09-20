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

#include <memory>

#include <QImage>
#include <QWidget>

#include "libvis/libvis.h"
#include "libvis/window_callbacks.h"

namespace vis {

// Qt widget for image display.
class ImageDisplayQtWidget : public QWidget
{
 Q_OBJECT
 public:
  ImageDisplayQtWidget(QWidget* parent = nullptr);
//   ImageDisplayQtWidget(const QImage& image, const shared_ptr<ImageWindowCallbacks>& callbacks, QWidget* parent = nullptr);
  ~ImageDisplayQtWidget();
  
  // Sets the image displayed by the widget.
  void SetImage(const QImage& image);
  void SetCallbacks(const shared_ptr<ImageWindowCallbacks>& callbacks);
  void SetViewOffset(double x, double y);
  void SetZoomFactor(double zoom_factor);
  
  void AddSubpixelDotPixelCornerConv(float x, float y, u8 r, u8 g, u8 b);
  void AddSubpixelLinePixelCornerConv(float x0, float y0, float x1, float y1, u8 r, u8 g, u8 b);
  void Clear();
  
  virtual QSize sizeHint() const override;
  
  inline double zoom_factor() const { return view_scale_; }
  inline const QImage& image() const { return image_; }
  
  inline const QTransform& image_to_viewport() const { return image_to_viewport_; }
  inline const QTransform& viewport_to_image() const { return viewport_to_image_; }

 signals:
  // (0, 0) is at the top-left corner of the image here.
  void CursorPositionChanged(QPointF pos, bool pixel_value_valid, QRgb pixel_value);
  
  void ZoomChanged(double zoom);

 protected:
  virtual void resizeEvent(QResizeEvent* event) override;
  virtual void paintEvent(QPaintEvent* event) override;
  virtual void mousePressEvent(QMouseEvent* event) override;
  virtual void mouseMoveEvent(QMouseEvent* event) override;
  virtual void mouseReleaseEvent(QMouseEvent* event) override;
  virtual void wheelEvent(QWheelEvent* event) override;
  virtual void keyPressEvent(QKeyEvent* event) override;
  virtual void keyReleaseEvent(QKeyEvent* event) override;

 private:
  struct SubpixelDot {
    QPointF xy;
    QRgb rgb;
  };
  
  struct SubpixelLine {
    QPointF xy0;
    QPointF xy1;
    QRgb rgb;
  };
  
  void UpdateViewTransforms();
  QPointF ViewportToImage(const QPointF& pos);
  QPointF ImageToViewport(const QPointF& pos);
  
  void startDragging(QPoint pos);
  void updateDragging(QPoint pos);
  void finishDragging(QPoint pos);
  
  // Transformations between viewport coordinates and image coordinates.
  // (0, 0) is at the top-left corner of the image here.
  QTransform image_to_viewport_;
  QTransform viewport_to_image_;
  
  // Mouse dragging handling.
  bool dragging_;
  QPoint drag_start_pos_;
  QCursor normal_cursor_;
  
  // View settings.
  double view_scale_;
  double view_offset_x_;
  double view_offset_y_;
  
  vector<SubpixelDot> dots_;
  vector<SubpixelLine> lines_;
  
  shared_ptr<ImageWindowCallbacks> callbacks_;
  
  QImage image_;
};

}
