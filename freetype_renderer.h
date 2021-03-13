#ifndef FREETYPE_RENDERER_H
#define REETYPE_RENDERER_H

#include <ft2build.h>
#include <freetype/freetype.h>
#include <freetype/ftoutln.h>


// To put text to image
class ft_renderer {
private:
  FT_Library lib_;
  FT_Face    face_;

  void stoc(int &c, int &i, const std::string &text) {
    unsigned char t1 = (unsigned char) text[i];
    if (t1 >= 0xF0) {
      c  = (((unsigned char)text[i] << 18) & 0x70000) |
        (((unsigned char)text[i+1] << 12) & 0xF000) |
        (((unsigned char)text[i+2] << 6) & 0x0FC0) |
        (((unsigned char)text[i+3] << 0) & 0x003F) ;
      i += 4;
    } else if (t1 >= 0xE0) {
      c  = (((unsigned char)text[i] << 12) & 0xF000) |
        (((unsigned char)text[i+1] << 6) & 0x0FC0) |
        (((unsigned char)text[i+2] << 0) & 0x003F) ;
      i += 3;
    } else if (t1 >= 0xC2) {
      c  = (((unsigned char)text[i] << 6) & 0x0FC0) |
        (((unsigned char)text[i+1] << 0) & 0x003F) ;
      i += 2;
    } else if (t1 > 0) {
      c  = text[i];
      i += 1;
    } else {
      c = '?';
      i += 1;
    }
  }

public:
  ft_renderer(std::vector<uint8_t>& fontdata) {
    FT_Init_FreeType(&lib_);
    FT_New_Memory_Face(lib_, 
        fontdata.data(), 
        fontdata.size(), 0, &face_); 
  }
  ~ft_renderer() {
    FT_Done_FreeType(lib_);
  }

public:
  void putText(cv::InputOutputArray _img, const std::string& text, cv::Point pos,
    double fontScale, cv::Scalar color, bool bottomLeftOrigin) {

    if (text.empty())
      return;

    FT_Set_Pixel_Sizes(face_, fontScale, fontScale);

    cv::Mat img = _img.getMat();

    for (int i = 0; text[i] != 0;) {
      int c;

      stoc(c, i, text);
      FT_Load_Char(face_, (FT_Long)c, 0); 
      FT_Render_Glyph(face_->glyph, FT_RENDER_MODE_MONO);
      FT_Bitmap *bmp = &(face_->glyph->bitmap);

      cv::Point loc = pos;
      loc.y = loc.y - (face_->glyph->metrics.horiBearingY >> 6) ;
      loc.x = loc.x + (face_->glyph->metrics.horiBearingX >> 6) ;
      int row, col, bit, cl;

      for (row = 0; row < bmp->rows; ++row) {
        if (loc.y + row > img.rows)
          continue;
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(loc.y + row);
        for (col = 0; col < bmp->pitch; ++col) {
          cl = bmp->buffer[ row * bmp->pitch + col];
          for (bit = 7; bit >= 0; --bit) {
            if (loc.x + col * 8 + (7 - bit) > img.cols)
              continue;
            if (((cl >> bit) & 0x01) == 1) {
              ptr[loc.x + col * 8 + (7 - bit)][0] = color[0];
              ptr[loc.x + col * 8 + (7 - bit)][1] = color[1];
              ptr[loc.x + col * 8 + (7 - bit)][2] = color[2];
            }
          }
        }
      }
      pos.x += (face_->glyph->advance.x) >> 6;
      pos.y += (face_->glyph->advance.y) >> 6;
    }
  }
};

#endif // REETYPE_RENDERER_H