//! The xclbin: the compiled design's container.
//!
//! # WHAT THIS FILE IS
//!
//! An xclbin is a file with a header, a list of sections, and the sections
//! themselves. The direct transport needs two of them:
//!
//! * the **AIE partition**, which holds the PDI -- the program the array
//!   runs;
//! * the **embedded metadata**, which holds the XML that describes the
//!   compute unit: its name, and where each of its arguments sits in the
//!   register map that the command packet carries.
//!
//! The XRT transport needs neither of them parsed: XRT parses the xclbin
//! itself, and hands the runtime a kernel it can call. The direct transport
//! has no XRT, so it reads the xclbin here -- and the reading is a PURE
//! FUNCTION of the bytes, which means it can be tested without a board.
//!
//! # THE FORMAT, AS THE FILES ON THIS BOARD HAVE IT
//!
//! The section header is forty bytes:
//!
//! ```text
//!  0  u32   the section's kind
//!  4  char  the section's name, sixteen bytes, NUL-padded
//! 20  u32   reserved
//! 24  u64   the section's offset in the file
//! 32  u64   the section's size in bytes
//! ```
//!
//! and the section headers themselves start at `0x1c8`, right after the
//! file header, whose last field is the section count at `0x1c0`.
//!
//! The AIE partition's own header is the one XRT's `xclbin.h` calls
//! `struct aie_partition`, and the PDI is found through it: the header
//! points at an array of `struct aie_pdi`, and each of those points at the
//! image. Every step of that is a field with a name, not a guess about
//! where a number might be.

/// The section kind of the embedded metadata: the XML with the kernel's
/// description.
const KIND_EMBEDDED_METADATA: u32 = 2;

/// The section kind of the AIE partition: the PDI.
const KIND_AIE_PARTITION: u32 = 32;

/// Where the section headers start, and where the section count is.
const SECTION_COUNT_OFFSET: usize = 0x1c0;
const SECTION_HEADERS_OFFSET: usize = 0x1c8;
const SECTION_HEADER_SIZE: usize = 40;

/// The size of `struct aie_partition` -- the AIE partition section's header.
const AIE_PARTITION_HEADER: usize = 184;

/// The size of `struct aie_pdi` -- one entry of the partition's PDI array.
const AIE_PDI_SIZE: usize = 96;

/// One compute unit's argument, as the metadata describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    /// The argument's name: `opcode`, `instr`, `ninstr`, `bo0` ...
    pub name: String,
    /// Where the argument sits in the register map.
    pub offset: u32,
    /// How big the argument is.
    pub size: u32,
}

/// The compute unit's description.
#[derive(Debug, Clone)]
pub struct Kernel {
    /// The kernel's name: `MLIR_AIE` for a design the AIE flow produced.
    pub name: String,
    /// The arguments, in the order the metadata lists them.
    pub args: Vec<Arg>,
}

impl Kernel {
    /// One argument, by name.
    pub fn arg(&self, name: &str) -> Option<&Arg> {
        self.args.iter().find(|a| a.name == name)
    }

    /// The register map's size in bytes.
    ///
    /// The map is the union of the arguments' ranges, and the packet's
    /// length is that union rounded up to a word. A design whose map has a
    /// hole in it would still be sent whole, because the packet's words are
    /// positions and not a list.
    pub fn map_size(&self) -> u32 {
        self.args
            .iter()
            .map(|a| a.offset.saturating_add(a.size))
            .max()
            .unwrap_or(0)
    }
}

/// One section of the xclbin.
#[derive(Debug, Clone, Copy)]
struct Section {
    kind: u32,
    offset: u64,
    size: u64,
}

/// A parsed xclbin.
#[derive(Debug, Clone)]
pub struct Xclbin {
    bytes: Vec<u8>,
    sections: Vec<Section>,
}

/// The AIE partition's description.
///
/// The two numbers the driver needs to make a context for this design: how
/// many columns the partition occupies, and how much work the array does
/// per cycle.
#[derive(Debug, Clone, Copy)]
pub struct Partition {
    /// How many columns the partition occupies.
    pub column_width: u32,
    /// The design's operations per cycle.
    pub operations_per_cycle: u32,
}

impl Xclbin {
    /// Read an xclbin.
    ///
    /// The bytes are copied: the transport keeps the xclbin's parts (the
    /// PDI) long after the caller has let go of the blob it came in.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        // The magic first: a file that is not an xclbin is not a short one
        // either, and "not an xclbin" is the answer the caller can use.
        if bytes.len() < 8 || &bytes[..8] != b"xclbin2\0" {
            return Err("the xclbin's magic is not xclbin2".to_string());
        }
        if bytes.len() < SECTION_HEADERS_OFFSET {
            return Err("the xclbin is shorter than its header".to_string());
        }
        let count = u32::from_le_bytes(
            bytes[SECTION_COUNT_OFFSET..SECTION_COUNT_OFFSET + 4]
                .try_into()
                .map_err(|_| "the xclbin's section count is unreadable".to_string())?,
        ) as usize;
        let end = SECTION_HEADERS_OFFSET + count * SECTION_HEADER_SIZE;
        if end > bytes.len() {
            return Err(format!(
                "the xclbin says it has {count} sections, and its headers do not fit in {} bytes",
                bytes.len()
            ));
        }
        let mut sections = Vec::with_capacity(count);
        for index in 0..count {
            let base = SECTION_HEADERS_OFFSET + index * SECTION_HEADER_SIZE;
            let kind = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let offset = u64::from_le_bytes(bytes[base + 24..base + 32].try_into().unwrap());
            let size = u64::from_le_bytes(bytes[base + 32..base + 40].try_into().unwrap());
            let end = offset.saturating_add(size);
            if end > bytes.len() as u64 {
                return Err(format!(
                    "the xclbin's section {index} runs past the end of the file"
                ));
            }
            sections.push(Section {
                kind,
                offset,
                size,
            });
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            sections,
        })
    }

    /// One section's bytes, by kind.
    pub fn section(&self, kind: u32) -> Option<&[u8]> {
        self.sections
            .iter()
            .find(|s| s.kind == kind)
            .map(|s| &self.bytes[s.offset as usize..(s.offset + s.size) as usize])
    }

    /// The AIE partition's description.
    ///
    /// The two numbers the driver needs to make a context for this design:
    /// how many columns the partition occupies, and how much work the array
    /// does per cycle.
    pub fn partition(&self) -> Result<Partition, String> {
        let section = self
            .section(KIND_AIE_PARTITION)
            .ok_or("the xclbin has no AIE partition")?;
        if section.len() < AIE_PARTITION_HEADER {
            return Err("the AIE partition's header is truncated".to_string());
        }
        Ok(Partition {
            operations_per_cycle: u32::from_le_bytes(section[8..12].try_into().unwrap()),
            column_width: u16::from_le_bytes(section[32..34].try_into().unwrap()) as u32,
        })
    }

    /// The PDI: the program the array runs.
    ///
    /// The AIE partition section's header points at an array of PDI
    /// descriptors, and each descriptor points at the image itself. The
    /// image is what the driver is given, byte for byte.
    pub fn pdi(&self) -> Result<&[u8], String> {
        let section = self
            .section(KIND_AIE_PARTITION)
            .ok_or("the xclbin has no AIE partition")?;
        if section.len() < AIE_PARTITION_HEADER {
            return Err("the AIE partition's header is truncated".to_string());
        }
        let count = u32::from_le_bytes(section[120..124].try_into().unwrap()) as usize;
        let array = u32::from_le_bytes(section[124..128].try_into().unwrap()) as usize;
        for index in 0..count {
            let base = array + index * AIE_PDI_SIZE;
            if base + AIE_PDI_SIZE > section.len() {
                break;
            }
            let size = u32::from_le_bytes(section[base + 16..base + 20].try_into().unwrap())
                as usize;
            let offset = u32::from_le_bytes(section[base + 20..base + 24].try_into().unwrap())
                as usize;
            let end = offset + size;
            if size > 0 && end <= section.len() {
                return Ok(&section[offset..end]);
            }
        }
        Err("the AIE partition has no PDI image".to_string())
    }

    /// The compute unit's description, from the embedded metadata.
    pub fn kernel(&self) -> Result<Kernel, String> {
        let xml = self
            .section(KIND_EMBEDDED_METADATA)
            .ok_or("the xclbin has no embedded metadata")?;
        parse_kernel(xml)
    }
}

/// Read the compute unit out of the metadata's XML.
///
/// The XML is not parsed as XML: it is read for the two things the packet
/// needs -- the kernel's name and its arguments' places in the register
/// map -- and a scanner that looks for those is a scanner that cannot be
/// broken by a namespace or a comment.
fn parse_kernel(xml: &[u8]) -> Result<Kernel, String> {
    let text = String::from_utf8_lossy(xml);
    let start = text
        .find("<kernel")
        .ok_or("the metadata has no kernel")?;
    let tag_end = text[start..]
        .find('>')
        .map(|i| start + i + 1)
        .ok_or("the metadata's kernel tag is truncated")?;
    let tag = &text[start..tag_end];
    let name = attribute(tag, "name").unwrap_or_default();

    let body_end = text[tag_end..]
        .find("</kernel>")
        .map(|i| tag_end + i)
        .unwrap_or(text.len());
    let body = &text[tag_end..body_end];

    let mut args = Vec::new();
    let mut cursor = 0usize;
    while let Some(found) = body[cursor..].find("<arg ") {
        let at = cursor + found;
        let close = body[at..]
            .find('>')
            .map(|i| at + i + 1)
            .unwrap_or(body.len());
        let arg = &body[at..close];
        args.push(Arg {
            name: attribute(arg, "name").unwrap_or_default(),
            offset: attribute(arg, "offset")
                .and_then(|v| parse_number(&v))
                .unwrap_or(0),
            size: attribute(arg, "size")
                .and_then(|v| parse_number(&v))
                .unwrap_or(0),
        });
        cursor = close;
    }
    if args.is_empty() {
        return Err("the metadata's kernel has no arguments".to_string());
    }
    Ok(Kernel { name, args })
}

/// One attribute's value, out of a tag's text.
fn attribute(tag: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let at = tag.find(&needle)? + needle.len();
    let end = tag[at..].find('"')? + at;
    Some(tag[at..end].to_string())
}

/// A number, in hexadecimal or in decimal.
fn parse_number(text: &str) -> Option<u32> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        text.parse::<u32>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A section header, as the files on this board have them.
    fn header(kind: u32, offset: u64, size: u64) -> Vec<u8> {
        let mut out = vec![0u8; SECTION_HEADER_SIZE];
        out[0..4].copy_from_slice(&kind.to_le_bytes());
        out[24..32].copy_from_slice(&offset.to_le_bytes());
        out[32..40].copy_from_slice(&size.to_le_bytes());
        out
    }

    /// An xclbin with the given sections.
    fn xclbin(sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = vec![0u8; SECTION_HEADERS_OFFSET];
        out[..8].copy_from_slice(b"xclbin2\0");
        out[SECTION_COUNT_OFFSET..SECTION_COUNT_OFFSET + 4]
            .copy_from_slice(&(sections.len() as u32).to_le_bytes());
        // The sections start after the header AND the section headers: an
        // offset that forgot the headers would point forty bytes into the
        // section before the one it names.
        let mut at = SECTION_HEADERS_OFFSET + sections.len() * SECTION_HEADER_SIZE;
        let mut body = Vec::new();
        for (kind, bytes) in sections {
            out.extend(header(*kind, at as u64, bytes.len() as u64));
            body.extend_from_slice(bytes);
            at += bytes.len();
        }
        out.extend(body);
        out
    }

    /// The PDI image's bytes, as the partition section holds them.
    fn partition(image: &[u8]) -> Vec<u8> {
        let mut section = vec![0u8; AIE_PARTITION_HEADER + AIE_PDI_SIZE];
        // The partition's header: the PDI array's count and offset.
        section[120..124].copy_from_slice(&1u32.to_le_bytes());
        section[124..128].copy_from_slice(&(AIE_PARTITION_HEADER as u32).to_le_bytes());
        // The PDI descriptor: the image's size and offset.
        let base = AIE_PARTITION_HEADER;
        let image_at = AIE_PARTITION_HEADER + AIE_PDI_SIZE;
        section[base + 16..base + 20].copy_from_slice(&(image.len() as u32).to_le_bytes());
        section[base + 20..base + 24].copy_from_slice(&(image_at as u32).to_le_bytes());
        section.extend_from_slice(image);
        section
    }

    #[test]
    fn the_magic_is_checked() {
        let error = Xclbin::parse(b"not an xclbin at all").unwrap_err();
        assert!(error.contains("magic"), "{error}");
    }

    #[test]
    fn the_pdi_is_found_through_the_partition() {
        let image = vec![0xa5u8; 96];
        let bytes = xclbin(&[(KIND_AIE_PARTITION, partition(&image))]);
        let xclbin = Xclbin::parse(&bytes).unwrap();
        assert_eq!(xclbin.pdi().unwrap(), image.as_slice());
    }

    #[test]
    fn the_kernel_is_read_out_of_the_metadata() {
        let xml = br#"<?xml version="1.0" encoding="utf-8"?>
<project>
  <platform>
    <device>
      <core>
        <kernel name="MLIR_AIE" language="c" type="dpu">
          <extended-data dpu_kernel_id="0x901" functional="0" subtype="1"/>
          <arg name="opcode" addressQualifier="0" id="0" size="0x8" offset="0x00" type="uint64_t"/>
          <arg name="instr" addressQualifier="1" id="1" size="0x8" offset="0x08" type="char *"/>
          <arg name="ninstr" addressQualifier="0" id="2" size="0x4" offset="0x10" type="uint32_t"/>
          <arg name="bo0" addressQualifier="1" id="3" size="0x8" offset="0x14" type="void*"/>
        </kernel>
      </core>
    </device>
  </platform>
</project>"#;
        let bytes = xclbin(&[(KIND_EMBEDDED_METADATA, xml.to_vec())]);
        let xclbin = Xclbin::parse(&bytes).unwrap();
        let kernel = xclbin.kernel().unwrap();
        assert_eq!(kernel.name, "MLIR_AIE");
        assert_eq!(kernel.args.len(), 4);
        assert_eq!(kernel.arg("bo0").unwrap().offset, 0x14);
        assert_eq!(kernel.arg("bo0").unwrap().size, 8);
        assert_eq!(kernel.arg("ninstr").unwrap().size, 4);
        // The map runs to the end of the last argument.
        assert_eq!(kernel.map_size(), 0x1c);
    }

    #[test]
    fn a_section_that_runs_past_the_file_is_refused() {
        let mut bytes = xclbin(&[(KIND_EMBEDDED_METADATA, vec![0u8; 16])]);
        // Make the size field bigger than the file.
        let base = SECTION_HEADERS_OFFSET;
        bytes[base + 32..base + 40].copy_from_slice(&4096u64.to_le_bytes());
        let error = Xclbin::parse(&bytes).unwrap_err();
        assert!(error.contains("past the end"), "{error}");
    }
}
