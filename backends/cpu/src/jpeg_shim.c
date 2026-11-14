// The image front end's JPEG decode, as a C shim.
//
// WHY A C SHIM AND NOT A RUST JPEG DECODER
// ----------------------------------------
// The classifier's answer is a probability printed to nine significant
// digits, and the golden transcript holds those digits. That number is the
// end of a chain that starts with PIXELS: a different IDCT gives different
// pixels, a different resize gives different pixels, and a different pixel
// is a different probability. So the decode cannot be "close enough" -- it
// has to be the same bytes.
//
// libjpeg's own IDCT is an INTEGER algorithm with a specified output, and
// re-implementing it in Rust would be re-implementing it: a hundred lines
// of exact integer arithmetic that has to agree with libjpeg's to the last
// bit, and a silent one-bit disagreement is a wrong probability that
// looks right. Calling the same library the oracle calls is not a
// shortcut here; it is the only way to be RIGHT.
//
// So this file is the C++ runtime's `decode_jpeg`, unchanged in what it
// asks libjpeg for, with the struct layout kept on the C side where
// libjpeg's headers define it. The Rust side sees three integers and a
// byte buffer.
//
// WHAT IT DOES NOT CHANGE
// -----------------------
// libjpeg's defaults are libjpeg's defaults: the "fancy" upsampling, the
// slow integer IDCT, the YCbCr -> RGB transform libjpeg ships. The C++
// runtime sets exactly one thing -- `out_color_space = JCS_RGB`, so a
// grayscale JPEG becomes three channels -- and so does this.

// stddef.h and stdio.h FIRST: jpeglib.h uses size_t and FILE without
// including them, and the include order is the whole difference between
// compiling and not.
#include <stddef.h>
#include <stdio.h>

#include <jpeglib.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

// The error handler's state. libjpeg reports a fatal error by longjmp'ing
// out of its own frames, so everything the handler needs has to exist
// BEFORE the setjmp: a local whose constructor has not run must not be
// destroyed on the way back.
struct npue_jpeg_error {
  struct jpeg_error_mgr pub;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};

static void npue_jpeg_error_exit(j_common_ptr cinfo) {
  struct npue_jpeg_error *err = (struct npue_jpeg_error *)cinfo->err;
  (*cinfo->err->format_message)(cinfo, err->message);
  longjmp(err->jump, 1);
}

// Decode a JPEG into RGB.
//
// Returns 1 on success, and fills `out` with a malloc'd buffer of
// `width * height * 3` bytes that the caller frees with `npue_free`.
// Returns 0 on failure, and fills `errbuf` with the reason.
int npue_jpeg_decode(const unsigned char *data, unsigned long length,
                     unsigned char **out, int *width, int *height,
                     char *errbuf, unsigned long errlen) {
  struct jpeg_decompress_struct cinfo;
  struct npue_jpeg_error jerr;
  // libjpeg's defaults, into a SEPARATE object, then copied:
  // jpeg_std_error writes six-odd function pointers and two counters
  // THROUGH the pointer it is given, so a self-assignment works and reads
  // as the accident it is.
  {
    struct jpeg_error_mgr defaults;
    jerr.pub = *jpeg_std_error(&defaults);
  }
  jerr.pub.error_exit = npue_jpeg_error_exit;
  jerr.message[0] = '\0';

  cinfo.err = &jerr.pub;
  jpeg_create_decompress(&cinfo);

  if (setjmp(jerr.jump)) {
    jpeg_destroy_decompress(&cinfo);
    snprintf(errbuf, (size_t)errlen, "%s",
             jerr.message[0] ? jerr.message : "JPEG decode failed");
    return 0;
  }

  jpeg_mem_src(&cinfo, data, length);
  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    jpeg_destroy_decompress(&cinfo);
    snprintf(errbuf, (size_t)errlen, "JPEG header is not readable");
    return 0;
  }

  // Grayscale and CMYK JPEGs to RGB, so the normalise step's per-channel
  // mean and std both exist. This is the ONE thing the C++ runtime sets,
  // and this sets the same one.
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);

  if (cinfo.output_components != 3) {
    jpeg_destroy_decompress(&cinfo);
    snprintf(errbuf, (size_t)errlen,
             "JPEG decodes to %d components; this runtime asked libjpeg for RGB",
             cinfo.output_components);
    return 0;
  }

  const size_t w = (size_t)cinfo.output_width;
  const size_t h = (size_t)cinfo.output_height;
  unsigned char *rgb = (unsigned char *)malloc(w * h * 3);
  if (!rgb) {
    jpeg_destroy_decompress(&cinfo);
    snprintf(errbuf, (size_t)errlen, "out of memory");
    return 0;
  }
  unsigned char *row = (unsigned char *)malloc(w * 3);
  if (!row) {
    free(rgb);
    jpeg_destroy_decompress(&cinfo);
    snprintf(errbuf, (size_t)errlen, "out of memory");
    return 0;
  }

  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW rp = row;
    if (jpeg_read_scanlines(&cinfo, &rp, 1) != 1) {
      free(row);
      free(rgb);
      jpeg_destroy_decompress(&cinfo);
      snprintf(errbuf, (size_t)errlen, "JPEG ended %lu of %lu lines",
               (unsigned long)cinfo.output_scanline,
               (unsigned long)cinfo.output_height);
      return 0;
    }
    memcpy(rgb + (size_t)(cinfo.output_scanline - 1) * w * 3, row, w * 3);
  }
  free(row);
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  *out = rgb;
  *width = (int)w;
  *height = (int)h;
  return 1;
}

// Free a buffer this file allocated.
void npue_free(void *p) { free(p); }
