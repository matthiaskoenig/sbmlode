"""Reading and flattening of SBML documents.

`read_document` reads a document from a path or an SBML string and raises for a
source which could not be read, `flatten` resolves the submodels of comp into one
model. Both are what the analysis needs of sbmlutils' `read_sbml` and
`flatten_sbml_doc`, on libsbml alone.
"""

import logging
from pathlib import Path

import libsbml

logger = logging.getLogger(__name__)

__all__ = ["flatten", "read_document"]

#: the errors libsbml reports on the XML declaration; the document itself is
#: read in full, only the declaration SBML requires is missing
_XML_DECLARATION_ERRORS: frozenset[int] = frozenset(
    {libsbml.MissingXMLDecl, libsbml.MissingXMLEncoding}
)


def _shorten(source: Path | str, limit: int = 200) -> str:
    """Shorten a source for a message, a string which is not a path can be a document."""
    text = str(source)
    return text if len(text) <= limit else text[:limit] + "..."


def _read(source: Path | str) -> tuple[libsbml.SBMLDocument, str]:
    """Read a document from a path or an SBML string without raising.

    libsbml passes a path to the narrow file API on Windows, which opens an ASCII
    path only for certain, so another path is read by python as UTF-8, the encoding
    SBML requires, with or without a byte order mark.

    Returns:
        the document with the errors libsbml reported while parsing it, and a label
        of the source for messages, which does not repeat an SBML string
    """
    if isinstance(source, str) and "<sbml" in source:
        return libsbml.readSBMLFromString(source), "SBML string"
    # libsbml resolves the `comp:source` of an external model definition against
    # the location of the document, which must be absolute
    path = Path(source).resolve()
    label = f"SBML file '{_shorten(path)}'"
    if str(path).isascii():
        return libsbml.readSBMLFromFile(str(path)), label
    doc: libsbml.SBMLDocument
    try:
        doc = libsbml.readSBMLFromString(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as err:
        doc = libsbml.SBMLDocument()
        doc.getErrorLog().add(
            libsbml.SBMLError(
                libsbml.XMLBadUTF8Content
                if isinstance(err, UnicodeDecodeError)
                else libsbml.XMLFileUnreadable,
                doc.getLevel(),
                doc.getVersion(),
                str(err),
            )
        )
    doc.setLocationURI(f"file:{path}")
    return doc, label


def _failures(doc: libsbml.SBMLDocument) -> list[libsbml.SBMLError]:
    """The errors which mean that a document could not be read.

    These are the errors of severity error or fatal which the file system or the
    XML parser reported: a file which cannot be opened, or content which is not
    well-formed XML. `BadXMLDecl` is a failure only if the document holds nothing,
    libsbml reports it as well for a file without an XML declaration which it reads
    in full.
    """
    comp_doc: libsbml.CompSBMLDocumentPlugin | None = doc.getPlugin("comp")
    was_read = doc.getModel() is not None or (
        comp_doc is not None and comp_doc.getNumModelDefinitions() > 0
    )
    failures: list[libsbml.SBMLError] = []
    for k in range(doc.getNumErrors()):
        error: libsbml.SBMLError = doc.getError(k)
        if (
            error.getSeverity() >= libsbml.LIBSBML_SEV_ERROR
            and error.getCategory()
            in (libsbml.LIBSBML_CAT_SYSTEM, libsbml.LIBSBML_CAT_XML)
            and error.getErrorId() not in _XML_DECLARATION_ERRORS
            and not (was_read and error.getErrorId() == libsbml.BadXMLDecl)
        ):
            failures.append(error)
    return failures


def read_document(source: Path | str) -> libsbml.SBMLDocument:
    """Read a document from a path or an SBML string.

    Args:
        source: the path of an SBML file or an SBML string

    Returns:
        the document; the errors of its content are in its error log, they are for
        a validation to report

    Raises:
        ValueError: if the source cannot be read, a file which cannot be opened or
            content which is not well-formed XML, with the errors libsbml reported
    """
    doc, label = _read(source)
    failures = _failures(doc)
    if failures:
        raise ValueError(
            f"{label} could not be read:\n"
            + "\n".join(
                f"  E{error.getErrorId()} ({error.getSeverityAsString()}): "
                f"{error.getMessage().strip()}"
                for error in failures
            )
        )
    return doc


def flatten(doc: libsbml.SBMLDocument) -> libsbml.SBMLDocument:
    """Flatten the submodels of comp into the model of a document, in place.

    Args:
        doc: the document, with a model

    Returns:
        the document

    Raises:
        ValueError: if the document has no model or cannot be flattened
    """
    if doc.getModel() is None:
        raise ValueError(
            "SBML without a model cannot be flattened, a document with only "
            "model definitions has no model to flatten them into."
        )
    props = libsbml.ConversionProperties()
    props.addOption("flatten comp", True)
    props.addOption("leave_ports", True)
    props.addOption("abortIfUnflattenable", "none")
    if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
        errors = "\n".join(
            f"  E{doc.getError(k).getErrorId()}: {doc.getError(k).getMessage().strip()}"
            for k in range(doc.getNumErrors())
            if doc.getError(k).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
        )
        raise ValueError(f"SBML could not be flattened due to errors:\n{errors}")
    return doc
