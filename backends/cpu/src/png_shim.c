// The image front end's PNG decode, as a C shim.
//
// WHY A C SHIM AND NOT A RUST PNG DECODER
// ---------------------------------------
// The OCR family's answer is a STRING, and the string is the end of a chain
// that starts with PIXELS: a different inflate, a different unfilter or a
// different gamma table gives different pixels, and a different pixel is a
// different transcription. So the decode cannot be "close enough" -- it has
// to be the same bytes.
//
// libpng's own inflate and unfilter are the reference implementation, and
// re-implementing them in Rust would be re-implementing them: hundreds of
// lines of exact arithmetic that has to agree with libpng's to the last bit,
// and a silent one-bit disagreement is a character that reads wrong inside
// a word that reads right. Calling the same library the oracle calls is not
// a shortcut here; it is the only way to be RIGHT.
//
// So this file is the reference runtime's `decode_png`, unchanged in what it asks
// libpng for, with the struct layout kept on the C side where libpng's
// headers define it. The Rust side sees three integers and a byte buffer.
//
// WHAT IT DOES NOT CHANGE
// -----------------------
// The same decisions the reference runtime makes, and for the same reasons:
//
//   * a PALETTE image becomes RGB (png_set_palette_to_rgb);
//   * a GRAY or GRAY_ALPHA image becomes RGB by REPLICATION
//     (png_set_gray_to_rgb), which is what PIL's convert("RGB") does and
//     therefore what the checkpoint was trained to see -- and NOT
//     png_set_add_alpha, which would yield a two-channel GA image that the
//     channel check then rejects;
//   * a tRNS key becomes a real alpha (png_set_tRNS_to_alpha);
//   * a FOUR-channel image is composited onto WHITE, because a viewer shows
//     a transparent PNG over whatever is behind it and in every document
//     and browser that is white.
//
// The 16-BIT refusal is NOT here: it is a policy decision about which
// wrong image is less wrong, and it belongs in the Rust caller, which can
// say it in the runtime's own words.

// stddef.h and stdio.h FIRST: png.h uses size_t and FILE without including
// them, and the include order is the whole difference between compiling and
// not.
#include <stddef.h>
#include <stdio.h>

#include <png.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

// The error handler's state. libpng reports a fatal error by longjmp'ing
// out of its own frames, so everything the handler needs has to exist
// BEFORE the setjmp.
struct npue_png_error {
  char message[256];
};

// The read source: a cursor into the caller's byte vector. It lives here,
// before the setjmp, because png_set_read_fn keeps the pointer for the rest
// of the decode and libpng calls back into it after a longjmp too.
struct npue_png_reader {
  const unsigned char *p;
  size_t n;
  size_t at;
};

static void npue_png_error_fn(png_structp p, png_const_charp msg) {
  struct npue_png_error *err = (struct npue_png_error *)png_get_error_ptr(p);
  if (err) {
    snprintf(err->message, sizeof(err->message), "%s", msg ? msg : "libpng error");
  }
  png_longjmp(p, 1);
}

static void npue_png_read_fn(png_structp p, png_bytep out, png_size_t want) {
  struct npue_png_reader *r = (struct npue_png_reader *)png_get_io_ptr(p);
  const size_t left = r->n - r->at;
  const size_t got = left < want ? left : want;
  memcpy(out, r->p + r->at, got);
  r->at += got;
  // A short read is png_error's business, not a silent zero: returning fewer
  // bytes than asked for without complaint is how a truncated PNG becomes a
  // half-grey image.
  if (got != want) png_error(p, "unexpected end of PNG data");
}

// Decode a PNG into RGB.
//
// Returns 1 on success, and fills `out` with a malloc'd buffer of
// `width * height * 3` bytes that the caller frees with `npue_free`.
// Returns 0 on failure, and fills `errbuf` with the reason.
//
// `bit_depth` is reported separately: the 16-bit refusal is the caller's,
// because the refusal is a policy and this file is a decoder.
int npue_png_decode(const unsigned char *data, unsigned long length,
                    unsigned char **out, int *width, int *height,
                    int *bit_depth, char *errbuf, unsigned long errlen) {
  if (png_sig_cmp((png_bytep)data, 0, 8) != 0) {
    snprintf(errbuf, (size_t)errlen, "not a PNG (bad signature)");
    return 0;
  }

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) {
    snprintf(errbuf, (size_t)errlen, "libpng: out of memory");
    return 0;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_read_struct(&png, NULL, NULL);
    snprintf(errbuf, (size_t)errlen, "libpng: out of memory");
    return 0;
  }

  struct npue_png_reader reader;
  reader.p = data;
  reader.n = (size_t)length;
  reader.at = 0;

  struct npue_png_error failure;
  failure.message[0] = '\0';

  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, NULL);
    snprintf(errbuf, (size_t)errlen, "%s",
             failure.message[0] ? failure.message : "PNG is corrupt or truncated");
    return 0;
  }

  png_set_error_fn(png, &failure, npue_png_error_fn, NULL);
  // WITHOUT A READ FUNCTION libpng falls back to png_init_io(), which calls
  // fread() on a FILE* it never got -- so the FIRST read dereferences null
  // and the process dies with SIGSEGV, on every PNG. A memory reader is the
  // whole reason the decoder takes a byte vector rather than a path.
  png_set_read_fn(png, &reader, npue_png_read_fn);

  png_read_info(png, info);

  png_uint_32 w = 0, h = 0;
  int depth = 0, colour = 0;
  png_get_IHDR(png, info, &w, &h, &depth, &colour, NULL, NULL, NULL);

  if (colour == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
  if (colour == PNG_COLOR_TYPE_GRAY || colour == PNG_COLOR_TYPE_GRAY_ALPHA) {
    if (depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    // Grayscale becomes RGB by REPLICATION, which is what PIL's convert("RGB")
    // does and therefore what the checkpoint was trained to see.
    //
    // It has to be png_set_gray_to_rgb and NOT png_set_add_alpha: adding an
    // opaque alpha to a GRAY image yields GA, TWO channels, and the channel
    // check below then rejects every grayscale PNG in the world.
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
      // tRNS on a gray image is a transparent-COLOR key. It has to become a
      // real alpha AFTER the gray has become RGB, or the key is expanded
      // against a two-channel image.
      png_set_gray_to_rgb(png);
      png_set_tRNS_to_alpha(png);
    } else {
      png_set_gray_to_rgb(png);
    }
  } else if (png_get_valid(png, info, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png);
  }
  png_set_interlace_handling(png);
  png_read_update_info(png, info);

  const int channels = png_get_channels(png, info);
  if (channels != 3 && channels != 4) {
    png_destroy_read_struct(&png, &info, NULL);
    snprintf(errbuf, (size_t)errlen,
             "PNG expands to %d channels, which is neither RGB nor RGBA", channels);
    return 0;
  }

  const size_t rowbytes = png_get_rowbytes(png, info);
  unsigned char *buf = (unsigned char *)malloc((size_t)h * rowbytes);
  if (!buf) {
    png_destroy_read_struct(&png, &info, NULL);
    snprintf(errbuf, (size_t)errlen, "out of memory");
    return 0;
  }
  png_bytep *rows = (png_bytep *)malloc((size_t)h * sizeof(png_bytep));
  if (!rows) {
    free(buf);
    png_destroy_read_struct(&png, &info, NULL);
    snprintf(errbuf, (size_t)errlen, "out of memory");
    return 0;
  }
  for (png_uint_32 y = 0; y < h; ++y) rows[y] = buf + (size_t)y * rowbytes;

  png_read_image(png, rows);
  png_read_end(png, NULL);
  free(rows);

  unsigned char *rgb = (unsigned char *)malloc((size_t)w * h * 3);
  if (!rgb) {
    free(buf);
    png_destroy_read_struct(&png, &info, NULL);
    snprintf(errbuf, (size_t)errlen, "out of memory");
    return 0;
  }
  for (png_uint_32 y = 0; y < h; ++y) {
    const unsigned char *s = buf + (size_t)y * rowbytes;
    unsigned char *d = rgb + (size_t)y * w * 3;
    if (channels == 3) {
      memcpy(d, s, (size_t)w * 3);
    } else {
      // Composite onto WHITE. A viewer shows a transparent PNG over whatever
      // is behind it, and in every document and browser that is white;
      // treating alpha as black instead silently turns a cut-out logo into a
      // dark rectangle and then reads it as one.
      for (png_uint_32 x = 0; x < w; ++x) {
        const unsigned a = s[(size_t)x * 4 + 3];
        for (int c = 0; c < 3; ++c) {
          const unsigned v = s[(size_t)x * 4 + c];
          d[x * 3 + c] = (unsigned char)((v * a + 255u * (255u - a)) / 255u);
        }
      }
    }
  }
  free(buf);
  png_destroy_read_struct(&png, &info, NULL);

  *out = rgb;
  *width = (int)w;
  *height = (int)h;
  *bit_depth = depth;
  return 1;
}
