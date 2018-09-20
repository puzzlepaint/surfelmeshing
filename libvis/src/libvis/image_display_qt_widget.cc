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


#include "libvis/image_display_qt_widget.h"

#include <cmath>

#include <QPainter>
#include <QPaintEvent>

namespace vis {

ImageDisplayQtWidget::ImageDisplayQtWidget(QWidget* parent)
    : QWidget(parent),
      callbacks_(nullptr) {
  dragging_ = false;
  
  view_scale_ = 1.0;
  view_offset_x_ = 0.0;
  view_offset_y_ = 0.0;
  image_ = QImage();
  UpdateViewTransforms();
  
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
  setMouseTracking(true);
  setFocusPolicy(Qt::ClickFocus);
  setSizePolicy(QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred));
}

ImageDisplayQtWidget::~ImageDisplayQtWidget() {}

void ImageDisplayQtWidget::SetImage(const QImage& image) {
  image_ = image;
  UpdateViewTransforms();
  update(rect());
}

void ImageDisplayQtWidget::SetCallbacks(const shared_ptr<ImageWindowCallbacks>& callbacks) {
  callbacks_ = callbacks;
  update(rect());
}

void ImageDisplayQtWidget::SetViewOffset(double x, double y) {
  view_offset_x_ = x;
  view_offset_y_ = y;
  UpdateViewTransforms();
  update(rect());
}

void ImageDisplayQtWidget::SetZoomFactor(double zoom_factor) {
  view_scale_ = zoom_factor;
  ZoomChanged(view_scale_);
  UpdateViewTransforms();
  update(rect());
}

void ImageDisplayQtWidget::AddSubpixelDotPixelCornerConv(float x, float y, u8 r, u8 g, u8 b) {
  dots_.emplace_back();
  SubpixelDot* new_dot = &dots_.back();
  new_dot->xy = QPointF(x, y);
  new_dot->rgb = qRgb(r, g, b);
}

void ImageDisplayQtWidget::AddSubpixelLinePixelCornerConv(float x0, float y0, float x1, float y1, u8 r, u8 g, u8 b) {
  lines_.emplace_back();
  SubpixelLine* new_line = &lines_.back();
  new_line->xy0 = QPointF(x0, y0);
  new_line->xy1 = QPointF(x1, y1);
  new_line->rgb = qRgb(r, g, b);
}

void ImageDisplayQtWidget::Clear() {
  dots_.clear();
  lines_.clear();
}

QSize ImageDisplayQtWidget::sizeHint() const {
  if (image_.isNull()) {
    return QSize(150, 150);
  } else {
    return image_.size();
  }
}

void ImageDisplayQtWidget::resizeEvent(QResizeEvent* event) {
  UpdateViewTransforms();
  QWidget::resizeEvent(event);
  if (callbacks_) {
    callbacks_->Resize(event->size().width(), event->size().height());
  }
}

void ImageDisplayQtWidget::paintEvent(QPaintEvent* event) {
  // Create painter and set its options.
  QPainter painter(this);
  QRect event_rect = event->rect();
  painter.setClipRect(event_rect);
  
  painter.fillRect(event_rect, QColor(Qt::gray));
  
  if (image_.isNull()) {
    return;
  }
  
  painter.setRenderHint(QPainter::Antialiasing, true);
  
  QTransform image_to_viewport_T = image_to_viewport_.transposed();
  painter.setTransform(image_to_viewport_T);
  
  painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
  painter.drawImage(QPointF(0, 0), image_);
  
  painter.resetTransform();
  
  if (!dots_.empty()) {
    painter.setBrush(Qt::NoBrush);
    
    for (const SubpixelDot& dot : dots_) {
      QPointF viewport_position = image_to_viewport_T.map(dot.xy);
      
      painter.setPen(dot.rgb);
      painter.drawEllipse(viewport_position, 2, 2);
    }
  }
  
  if (!lines_.empty()) {
    painter.setBrush(Qt::NoBrush);
    
    for (const SubpixelLine& line : lines_) {
      QPointF viewport_position_0 = image_to_viewport_T.map(line.xy0);
      QPointF viewport_position_1 = image_to_viewport_T.map(line.xy1);
      
      painter.setPen(line.rgb);
      painter.drawLine(viewport_position_0, viewport_position_1);
    }
  }
  
  if (callbacks_) {
    callbacks_->Render(&painter);
  }
  
  painter.end();
}

void ImageDisplayQtWidget::mousePressEvent(QMouseEvent* event) {
  if (dragging_) {
    event->accept();
    return;
  }
  
  ImageWindowCallbacks::MouseButton button;
  if (event->button() == Qt::LeftButton) {
    button = ImageWindowCallbacks::MouseButton::kLeft;
  } else if (event->button() == Qt::MiddleButton) {
    button = ImageWindowCallbacks::MouseButton::kMiddle;
  } else if (event->button() == Qt::RightButton) {
    button = ImageWindowCallbacks::MouseButton::kRight;
  } else {
    return;
  }
  
  if (callbacks_) {
    callbacks_->MouseDown(button, event->pos().x(), event->pos().y());
  }

  if (event->button() == Qt::MiddleButton) {
    startDragging(event->pos());
    event->accept();
  }
}

void ImageDisplayQtWidget::mouseMoveEvent(QMouseEvent* event) {
  QPointF image_pos = ViewportToImage(event->localPos());
  QRgb pixel_value = 0;
  bool pixel_value_valid =
      !image_.isNull() && image_pos.x() >= 0 && image_pos.y() >= 0 &&
      image_pos.x() < image_.width() && image_pos.y() < image_.height();
  if (pixel_value_valid) {
    int x = image_pos.x();
    int y = image_pos.y();
    pixel_value = image_.pixel(x, y);
  }
  emit CursorPositionChanged(image_pos, pixel_value_valid, pixel_value);
  
  if (dragging_) {
    updateDragging(event->pos());
    return;
  }
  
  if (callbacks_) {
    callbacks_->MouseMove(event->pos().x(), event->pos().y());
  }
}

void ImageDisplayQtWidget::mouseReleaseEvent(QMouseEvent* event) {
  if (dragging_) {
    finishDragging(event->pos());
    event->accept();
    return;
  }
  
  ImageWindowCallbacks::MouseButton button;
  if (event->button() == Qt::LeftButton) {
    button = ImageWindowCallbacks::MouseButton::kLeft;
  } else if (event->button() == Qt::MiddleButton) {
    button = ImageWindowCallbacks::MouseButton::kMiddle;
  } else if (event->button() == Qt::RightButton) {
    button = ImageWindowCallbacks::MouseButton::kRight;
  } else {
    return;
  }
  
  if (callbacks_) {
    callbacks_->MouseUp(button, event->pos().x(), event->pos().y());
  }
  event->accept();
}

void ImageDisplayQtWidget::wheelEvent(QWheelEvent* event) {
  if (event->orientation() == Qt::Vertical) {
    double degrees = event->delta() / 8.0;
    double num_steps = degrees / 15.0;
    
    double scale_factor = pow(sqrt(2.0), num_steps);
    
    // viewport_to_image_.m11() * pos.x() + viewport_to_image_.m13() == (pos.x() - (new_view_offset_x_ + (0.5 * width()) - (0.5 * image_.width()) * new_view_scale_)) / new_view_scale_;
    QPointF center_on_image = ViewportToImage(event->pos());
    view_offset_x_ = event->pos().x() - (0.5 * width() - (0.5 * image_.width()) * (view_scale_ * scale_factor)) - (view_scale_ * scale_factor) * center_on_image.x();
    view_offset_y_ = event->pos().y() - (0.5 * height() - (0.5 * image_.height()) * (view_scale_ * scale_factor)) - (view_scale_ * scale_factor) * center_on_image.y();
    view_scale_ = view_scale_ * scale_factor;
    emit ZoomChanged(view_scale_);
    
    UpdateViewTransforms();
    
    if (callbacks_ && event->orientation() == Qt::Vertical) {
      callbacks_->WheelRotated(event->delta() / 8.0f, ImageWindowCallbacks::ConvertQtModifiers(event));
    }
    
    update(rect());
  } else {
    event->ignore();
  }
}

void ImageDisplayQtWidget::keyPressEvent(QKeyEvent* event) {
  if (callbacks_ && event->text().size() > 0) {
    callbacks_->KeyPressed(event->text()[0].toLatin1(), ImageWindowCallbacks::ConvertQtModifiers(event));
  }
}

void ImageDisplayQtWidget::keyReleaseEvent(QKeyEvent* event) {
  if (callbacks_ && event->text().size() > 0) {
    callbacks_->KeyReleased(event->text()[0].toLatin1(), ImageWindowCallbacks::ConvertQtModifiers(event));
  }
}

void ImageDisplayQtWidget::UpdateViewTransforms() {
  image_to_viewport_.setMatrix(
      view_scale_,           0,   view_offset_x_ + (0.5 * width()) - (0.5 * image_.width()) * view_scale_,
                0, view_scale_, view_offset_y_ + (0.5 * height()) - (0.5 * image_.height()) * view_scale_,
                0,           0,                                                                         1);
  viewport_to_image_ = image_to_viewport_.inverted();
}

QPointF ImageDisplayQtWidget::ViewportToImage(const QPointF& pos) {
  return QPointF(viewport_to_image_.m11() * pos.x() + viewport_to_image_.m12() * pos.y() + viewport_to_image_.m13(),
                 viewport_to_image_.m21() * pos.x() + viewport_to_image_.m22() * pos.y() + viewport_to_image_.m23());
}

QPointF ImageDisplayQtWidget::ImageToViewport(const QPointF& pos) {
  return QPointF(image_to_viewport_.m11() * pos.x() + image_to_viewport_.m12() * pos.y() + image_to_viewport_.m13(),
                 image_to_viewport_.m21() * pos.x() + image_to_viewport_.m22() * pos.y() + image_to_viewport_.m23());
}

void ImageDisplayQtWidget::startDragging(QPoint pos) {
//   Q_ASSERT(!dragging);
  dragging_ = true;
  drag_start_pos_ = pos;
  normal_cursor_  = cursor();
  setCursor(Qt::ClosedHandCursor);
}

void ImageDisplayQtWidget::updateDragging(QPoint pos) {
//   Q_ASSERT(dragging);
  view_offset_x_ += (pos - drag_start_pos_).x();
  view_offset_y_ += (pos - drag_start_pos_).y();
  drag_start_pos_ = pos;
  UpdateViewTransforms();
  update(rect());
}

void ImageDisplayQtWidget::finishDragging(QPoint pos) {
//   Q_ASSERT(dragging);
  view_offset_x_ += (pos - drag_start_pos_).x();
  view_offset_y_ += (pos - drag_start_pos_).y();
  drag_start_pos_ = pos;
  UpdateViewTransforms();
  update(rect());
  
  dragging_ = false;
  setCursor(normal_cursor_);
}
}
