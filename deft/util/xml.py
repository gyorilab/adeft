from lxml import etree
from indra.literature.elsevier_client import extract_text, logger


logger.disabled = True


def _get_plaintext_pubmed(xml, strip_references=True):
    tree = etree.fromstring(xml)
    # remove references
    if strip_references:
        for xref in tree.findall('//xref'):
            xref.getparent().remove(xref)
    # get paragraphs
    paragraphs = [p.text for p in tree.xpath('//p') if p.text]
    paragraphs = ' '.join(paragraphs)
    paragraphs = ' '.join(paragraphs.split())
    return paragraphs


def to_plaintext(xml):
    try:
        result = extract_text(xml)
    except Exception:
        result = None
    if not result:
        try:
            result = _get_plaintext_pubmed(xml)
        except Exception:
            result = xml
    return result
