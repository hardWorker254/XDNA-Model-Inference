//! Just enough multipart/form-data to read a request body.
//!
//! # WHY A PARSER AND NOT A REGEX
//!
//! The body is BINARY: a WAV file is inside it, and it can contain the
//! boundary string by accident, `\r\n--boundary-` sequences, and NUL bytes.
//! A regex over it is a guess about bytes, and a guess about the bytes of
//! an audio file is how a transcription silently reads the wrong audio. So
//! this walks the body the way RFC 7578 says a parser must: find the first
//! boundary as a line, then for each part read the headers up to the blank
//! line, then take EXACTLY the number of bytes the headers say, then look
//! for the CRLF that precedes the next boundary.
//!
//! # WHAT IT REFUSES
//!
//! A missing boundary in the Content-Type, a part with no blank line after
//! its headers, a truncated body, a header block over 8 KB, and a boundary
//! over 70 characters. Each is an error with a reason, not a best-effort
//! parse: the alternatives are a wrong transcript and a wrong transcript.

/// One part of a multipart body.
#[derive(Debug, Clone)]
pub struct Part {
    /// The form field's name.
    pub name: String,
    /// Empty unless the part carried one.
    pub filename: String,
    /// Empty unless the part declared one.
    pub content_type: String,
    /// The bytes, verbatim.
    pub data: Vec<u8>,
}

/// The boundary out of a Content-Type header, or `None` when there is not
/// exactly one.
///
/// `multipart/form-data; boundary=----WebKitFormBoundaryABC` and the
/// quoted spelling are both accepted, because clients send both.
///
/// The KEY is matched case-insensitively (HTTP header names are), but the
/// VALUE is taken from the original header and NOT lowercased: a boundary
/// is an opaque byte string, and a boundary containing an uppercase letter
/// is a boundary the body will not contain once the header has been
/// folded.
pub fn boundary(content_type: &str) -> Option<String> {
    let lower = content_type.to_ascii_lowercase();
    if !lower.starts_with("multipart/form-data") {
        return None;
    }
    let at = lower.find("boundary=")?;
    // Same offset in both strings: lowercasing does not move characters.
    let mut i = at + 9;
    let bytes = content_type.as_bytes();
    if i < bytes.len() && bytes[i] == b'"' {
        i += 1;
        let end = content_type[i..].find('"')? + i;
        if end - i > 70 {
            return None;
        }
        return Some(content_type[i..end].to_string());
    }
    let end = content_type[i..]
        .find(|c| c == ';' || c == ' ' || c == '\r' || c == '\n')
        .map(|e| e + i)
        .unwrap_or(content_type.len());
    let b = &content_type[i..end];
    if b.is_empty() || b.len() > 70 {
        return None;
    }
    Some(b.to_string())
}

/// Header value out of one part's header block, lowercased key.
fn part_header(headers: &str, name: &str) -> String {
    let want = name.to_ascii_lowercase();
    let mut pos = 0usize;
    while pos < headers.len() {
        let eol = match headers[pos..].find("\r\n") {
            Some(e) => pos + e,
            None => headers.len(),
        };
        let line = &headers[pos..eol];
        if let Some(colon) = line.find(':') {
            let key = line[..colon].to_ascii_lowercase();
            let mut v = colon + 1;
            let bytes = line.as_bytes();
            while v < line.len() && (bytes[v] == b' ' || bytes[v] == b'\t') {
                v += 1;
            }
            if key == want {
                return line[v..].to_string();
            }
        }
        pos = eol + 2;
    }
    String::new()
}

/// Parse a multipart body.
///
/// `err` receives the reason when the parse fails.
pub fn parse(body: &[u8], boundary: &str) -> Result<Vec<Part>, String> {
    let dash = format!("--{boundary}");
    let dash_bytes = dash.as_bytes();
    // A body that does not even open with the first boundary is not this
    // media type; saying so is more useful than returning zero parts.
    let mut pos = match find(body, dash_bytes, 0) {
        Some(at) => at + dash_bytes.len(),
        None => {
            return Err(format!(
                "the body does not contain the boundary {dash} -- this does not \
                 look like multipart/form-data"
            ))
        }
    };
    let mut out: Vec<Part> = Vec::new();
    loop {
        // "--" after the boundary closes the body; a trailing "--" may follow.
        if body.len() >= pos + 2 && &body[pos..pos + 2] == b"--" {
            return Ok(out);
        }
        if body.len() < pos + 2 || &body[pos..pos + 2] != b"\r\n" {
            return Err(format!(
                "malformed boundary at byte {pos}: expected CRLF after {dash}"
            ));
        }
        pos += 2;
        let head_end = match find(body, b"\r\n\r\n", pos) {
            Some(at) => at,
            None => return Err("a part's headers are not terminated by a blank line".to_string()),
        };
        let headers = String::from_utf8_lossy(&body[pos..head_end]).into_owned();
        if headers.len() > 8192 {
            return Err("a part's header block is over 8 KB".to_string());
        }
        pos = head_end + 4;

        let mut part = Part {
            name: String::new(),
            filename: String::new(),
            content_type: String::new(),
            data: Vec::new(),
        };
        // Content-Disposition carries both the field name and the filename,
        // quoted.
        let disp = part_header(&headers, "content-disposition");
        match quoted(&disp, "name") {
            Some(name) => part.name = name,
            None => return Err("a part has no name= in its Content-Disposition".to_string()),
        }
        if let Some(filename) = quoted(&disp, "filename") {
            part.filename = filename;
        }
        part.content_type = part_header(&headers, "content-type");

        // The part ends at the CRLF BEFORE the next boundary, or at the
        // closing "--" for the last one. Anything else and a WAV containing
        // the boundary's first bytes would end the part early.
        let next = match find(body, format!("\r\n{dash}").as_bytes(), pos) {
            Some(at) => at,
            None => {
                return Err(format!(
                    "the body ends inside a part (no closing boundary after {})",
                    part.name
                ))
            }
        };
        part.data = body[pos..next].to_vec();
        out.push(part);
        pos = next + 2 + dash_bytes.len();
    }
}

/// A quoted value out of a header, by its key.
fn quoted(header: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let at = header.find(&needle)? + needle.len();
    let end = header[at..].find('"')? + at;
    Some(header[at..end].to_string())
}

/// The first occurrence of `needle` in `haystack` at or after `from`.
fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from > haystack.len() {
        return None;
    }
    if haystack.len() - from < needle.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|at| at + from)
}
