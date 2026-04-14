from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from textwrap import dedent
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def split_paragraphs(text: str) -> list[str]:
    block = dedent(text).strip()
    if not block:
        return []
    parts = re.split(r"\n\s*\n", block)
    cleaned: list[str] = []
    for part in parts:
        lines = [line.strip() for line in part.splitlines()]
        paragraph = " ".join(line for line in lines if line)
        if paragraph:
            cleaned.append(paragraph)
    return cleaned


def extract_block(path: Path, start_marker: str, end_marker: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_index = next(
        (idx for idx, line in enumerate(lines) if start_marker in line),
        None,
    )
    if start_index is None:
        raise ValueError(f"Could not find start marker in {path}: {start_marker}")

    end_index = next(
        (idx for idx in range(start_index, len(lines)) if end_marker in lines[idx]),
        None,
    )
    if end_index is None:
        raise ValueError(f"Could not find end marker in {path}: {end_marker}")
    return "\n".join(lines[start_index : end_index + 1]).rstrip()


class DocxBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.body_word_count = 0
        self.code_line_count = 0
        self.page_breaks = 0

    def _paragraph(self, text: str, style: str, *, preserve: bool = False) -> None:
        safe = escape(text if text else " ")
        space_attr = ' xml:space="preserve"' if preserve or safe.startswith(" ") or safe.endswith(" ") else ""
        self.parts.append(
            f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
            f'<w:r><w:t{space_attr}>{safe}</w:t></w:r></w:p>'
        )

    def add_title(self, text: str) -> None:
        self._paragraph(text, "Title")

    def add_subtitle(self, text: str) -> None:
        self._paragraph(text, "Subtitle")

    def add_heading(self, text: str, level: int = 1) -> None:
        self._paragraph(text, f"Heading{level}")

    def add_paragraph(self, text: str) -> None:
        self.body_word_count += len(re.findall(r"\b\w+\b", text))
        self._paragraph(text, "BodyText")

    def add_paragraphs(self, text: str) -> None:
        for paragraph in split_paragraphs(text):
            self.add_paragraph(paragraph)

    def add_code_block(self, text: str) -> None:
        lines = text.rstrip().splitlines() or [""]
        for line in lines:
            self.code_line_count += 1
            self._paragraph(line if line else " ", "CodeBlock", preserve=True)

    def add_page_break(self) -> None:
        self.page_breaks += 1
        self.parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def estimate_pages(self) -> int:
        text_pages = self.body_word_count / 360.0
        code_pages = self.code_line_count / 38.0
        structural_pages = max(self.page_breaks - 1, 0)
        return max(1, round(text_pages + code_pages + structural_pages))

    def document_xml(self) -> str:
        body = "".join(self.parts)
        sect_pr = (
            "<w:sectPr>"
            '<w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="720" w:footer="720" w:gutter="0"/>'
            "</w:sectPr>"
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}">'
            f"<w:body>{body}{sect_pr}</w:body></w:document>"
        )


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
        <w:lang w:val="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="140" w:line="300" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="BodyText">
    <w:name w:val="Body Text"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="160" w:line="300" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="200" w:after="220"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="34"/>
      <w:szCs w:val="34"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:i/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="240" w:after="140"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="30"/>
      <w:szCs w:val="30"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="180" w:after="100"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="26"/>
      <w:szCs w:val="26"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="140" w:after="80"/>
      <w:keepNext/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock">
    <w:name w:val="Code Block"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:ind w:left="300" w:right="180"/>
      <w:spacing w:after="0" w:line="240" w:lineRule="auto"/>
      <w:shd w:val="clear" w:color="auto" w:fill="F3F3F3"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>
      <w:sz w:val="18"/>
      <w:szCs w:val="18"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def settings_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
  <w:hyphenationZone w:val="425"/>
</w:settings>
"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company>OpenAI</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>
"""


def core_xml() -> str:
    stamp = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Expanded AI Forecasting Project Report</dc:title>
  <dc:subject>Detailed rewrite of PROJECT_LOG_FULL</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:keywords>forecasting, pytorch, tensorflow, migration, report</cp:keywords>
  <dc:description>Expanded Word report generated from PROJECT_LOG_FULL.md and source files.</dc:description>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{stamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{stamp}</dcterms:modified>
</cp:coreProperties>
"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>
"""


def add_comparison(
    doc: DocxBuilder,
    title: str,
    before_label: str,
    before_code: str,
    after_label: str,
    after_code: str,
    commentary: str,
) -> None:
    doc.add_heading(title, level=2)
    doc.add_heading(before_label, level=3)
    doc.add_code_block(before_code)
    doc.add_heading(after_label, level=3)
    doc.add_code_block(after_code)
    doc.add_paragraphs(commentary)


def add_timeline(doc: DocxBuilder, title: str, text: str) -> None:
    doc.add_heading(title, level=2)
    doc.add_paragraphs(text)


def collect_snippets(rewrite_root: Path) -> dict[str, str]:
    p1 = rewrite_root / "learning-path" / "01-tensorflow-basics" / "ai_predict.py"
    p3 = rewrite_root / "learning-path" / "03-tensorflow-dynamic-dropout" / "test_modified.py"
    p4 = rewrite_root / "learning-path" / "04-tensorflow-extended-forecasting" / "future-Eric.py"
    p5 = rewrite_root / "learning-path" / "05-pytorch-transition" / "future-eric-pytorch.py"
    p6 = rewrite_root / "learning-path" / "06-pytorch-stable-core" / "pytorch_fixed.py"
    p7 = rewrite_root / "learning-path" / "07-pytorch-full-pipeline" / "pytorch_plotted.py"
    p8 = rewrite_root / "learning-path" / "08-pytorch-experiments" / "pytorch_additional.py"
    p9t = rewrite_root / "separated" / "pytorch_train_cpp.py"
    p9p = rewrite_root / "separated" / "pytorch_plot_cpp.py"

    return {
        "phase1_core": extract_block(
            p1,
            "scaler = MinMaxScaler(feature_range=(0, 1))",
            "model.fit(x_train, y_train, epochs=10, batch_size=10000000)",
        ),
        "phase1_leakage": extract_block(
            p1,
            "model_inputs = model_inputs.reshape(-1, 1)",
            "model_inputs = scaler.fit_transform(model_inputs)",
        ),
        "phase1_model": extract_block(
            p1,
            "model = Sequential()",
            "model.add(Dense(units=1))",
        ),
        "phase3_dynamic": extract_block(
            p3,
            "def get_dynamic_dropout(epoch, total_epochs, initial_rate=0.5, final_rate=0.1):",
            "dynamic_dropout = DynamicDropoutCallback(epochs, initial_dropout, final_dropout)",
        ),
        "phase4_features": extract_block(
            p4,
            "df['returns'] = df['log'].diff()",
            "scaled_features_train = np.hstack([scaled_returns, scaled_X_train])",
        ),
        "phase4_residual": extract_block(
            p4,
            "early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)",
            "gbr_res.fit(x_train_flat, residuals)",
        ),
        "phase5_device": extract_block(
            p5,
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "y_train_tensor = torch.FloatTensor(y_train).to(device)",
        ),
        "phase5_model": extract_block(
            p5,
            "class DynamicLSTM(nn.Module):",
            "        return out",
        ),
        "phase6_model": extract_block(
            p6,
            "class LSTMModel(nn.Module):",
            "        return out",
        ),
        "phase6_training": extract_block(
            p6,
            "criterion = nn.HuberLoss()",
            "            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
        ),
        "phase7_backend": extract_block(
            p7,
            "torch.backends.cudnn.enabled = torch.version.hip is None",
            'os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"  # RDNA4-specific scheduling',
        ),
        "phase7_amp": extract_block(
            p7,
            "optimizer = getattr(optim, optimizer_name)(model.parameters(), weight_decay=0.05)",
            "amp_scaler.update()",
        ),
        "phase7_mc": extract_block(
            p7,
            "# Use model.train() instead of eval() to keep dropout active for jagged (stochastic) predictions",
            "future_predictions_std.append(",
        ),
        "phase8_attention": extract_block(
            p8,
            "class EnhancedLSTMModel(nn.Module):",
            "        return out",
        ),
        "phase9_paths": extract_block(
            p9t,
            "SCRIPT_DIR = Path(__file__).resolve().parent",
            "    return artifact_dirs",
        ),
        "phase9_vector": extract_block(
            p9t,
            "def build_sequences(values, prediction_days):",
            "    return x, y",
        ),
        "phase9_compile": extract_block(
            p9t,
            "    if use_compile:",
            '            print(f"torch.compile skipped: {e}")',
        ),
        "phase9_plot_discovery": extract_block(
            p9p,
            'DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_ROOT / "predictions"',
            "    return max(candidates, key=os.path.getmtime)",
        ),
    }


def write_docx(output_path: Path, builder: DocxBuilder) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", package_rels_xml())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("word/document.xml", builder.document_xml())
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/settings.xml", settings_xml())
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml())


def build_report() -> DocxBuilder:
    docs_dir = Path(__file__).resolve().parent
    rewrite_root = docs_dir.parent
    snippets = collect_snippets(rewrite_root)
    today = dt.datetime.now().strftime("%d %B %Y")

    doc = DocxBuilder()
    doc.add_title("Expanded AI Forecasting Project Report")
    doc.add_subtitle("A long-form Word rewrite of PROJECT_LOG_FULL.md with narrative explanations and representative code comparisons")
    doc.add_subtitle(f"Prepared from repository sources on {today}")

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraphs(
        """
        This report expands the repository's `PROJECT_LOG_FULL.md` into a much longer project narrative that reads like a formal engineering document rather than a compact migration note. The goal is not simply to restate the folder order. The goal is to explain why each stage existed, what technical pressure created the next rewrite, which design choices became permanent, and where the most meaningful code transitions can be seen in the actual files.

        Across the whole history, the project follows a clear pattern. A script begins life as a direct solution to a narrow problem, usually proving that a prediction can be produced at all. That first success then reveals a deeper weakness: weak features, scaler misuse, poor hardware behavior, smoothed-out forecasts, or maintenance pain caused by giant scripts. Each subsequent phase therefore solves both an ML problem and a software-engineering problem. The machine-learning side asks how to improve forecasting realism and stability; the engineering side asks how to keep experimentation possible as the code becomes more ambitious.

        The early TensorFlow work established the project's durable backbone: download market data, normalize it, build rolling windows, train an LSTM, and compare actual values against predicted ones on a chart. The later PyTorch work did not discard that backbone. Instead, it made the same basic workflow more explicit, more hardware-aware, and more modular. By the time the separated `cpp`-style pipeline appears, the code has moved from a proof-of-concept script into something much closer to a repeatable forecasting workstation with stable artifact output and reusable visualizations.

        The document also includes representative before-and-after code exhibits. These are intentionally drawn from the real source files rather than invented pseudocode. In some cases the project did not evolve by one clean commit replacing one function with another, so the exhibits should be read as architectural comparisons between stages rather than literal line-for-line diffs. That distinction matters because this repository evolved organically over time, and the most important change was often a shift in structure or discipline rather than a tiny syntactic edit.
        """
    )

    doc.add_heading("Method and Scope", level=1)
    doc.add_paragraphs(
        """
        The report is grounded in the current repository, especially the staged folders under `REWRITE/learning-path/`, the separated scripts under `REWRITE/separated/`, and the existing documentation in `REWRITE/docs/`. The narrative is therefore anchored in surviving code, comments, and file organization rather than reconstructed from memory alone. That makes the explanations more reliable, because each claim can be traced back to a concrete artifact that still exists in the tree.

        A second important scope decision is that the document treats the project as both a forecasting effort and an iterative software system. A short log can afford to say that a feature was added. A long engineering report needs to say what the feature changed in practice. For example, adding `HuberLoss` is not just a new import. It is a decision about how the system should react to financial outliers. Likewise, splitting the plotting logic into its own script is not just housekeeping. It changes how quickly someone can iterate, how artifacts are reproduced, and how failure modes are isolated.

        Finally, the report is written for a reader who may not have lived through each rewrite. That means the emphasis stays on continuity. Rather than presenting each folder as a separate mini-project, the document repeatedly tracks the same recurring concerns: sequence generation, feature engineering, regularization, device control, uncertainty estimation, experiment tracking, and operational maintainability. Seeing those threads carry forward is the easiest way to understand why the current architecture looks the way it does.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 1: TensorFlow Basics", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        This baseline phase is anchored to the earliest surviving repository import on December 18, 2025, with local file timestamps on `ai_predict.py` showing December 12, 2025 as the original script-writing period. In practical terms, that means the close-only TensorFlow predictor belongs to mid-December 2025, before the project had adopted richer features, explicit validation discipline, or hardware-aware PyTorch control.

        The key dated change in this phase was simply that the project became real. By December 18, 2025, the repository already contained a functioning LSTM forecasting script capable of downloading price data, preparing windows, training a stacked recurrent model, and exporting plots. That date matters because every later phase can be read as a reaction to the constraints that this initial December 2025 baseline exposed.
        """,
    )
    doc.add_paragraphs(
        """
        The first stage of the project proved that the end-to-end concept worked. Data could be downloaded from `yfinance`, the close-price series could be scaled, rolling windows could be built, and an LSTM could generate next-step predictions that were at least directionally meaningful. This is the stage where the system learned what its own minimum viable shape was. Even though the later code became far more sophisticated, the essential pattern of windowed sequence forecasting was already present here.

        What makes this phase important is not elegance but decisiveness. Many forecasting ideas never leave the notebook stage because the author spends too long optimizing architecture before establishing a working pipeline. These early scripts did the opposite. They accepted a simplistic feature space and a tightly coupled script in exchange for something much more valuable at the time: a concrete baseline. Once actual and predicted prices could be plotted on the same chart, every later change had a visible benchmark to improve against.

        The limitations of the phase are equally instructive. The model only sees `Close`, which means it has no direct representation of momentum, volatility, participation, or calendar effects. The script also mixes downloading, preprocessing, training, evaluation, plotting, and export into one continuous block. That is acceptable for a first prototype, but it quickly becomes painful when a developer wants to adjust only one concern. The most subtle weakness is the scaler discipline: fitting on combined inputs during test preparation risks leaking future distribution information into the evaluation process.

        Even so, the first phase deserves to be treated as foundational rather than naive. It established a vocabulary that continues all the way into the PyTorch era: `prediction_days`, sliding windows, inverse scaling, future rollout, and chart-based evaluation. Every later improvement can be understood as a response to something that this phase successfully surfaced. In that sense, the first scripts did exactly what good prototypes should do. They worked just well enough to expose what mattered next.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 1: The Original Close-Only Pipeline",
        "Original TensorFlow Training Flow",
        snippets["phase1_core"],
        "Original Fixed-Dropout Model Definition",
        snippets["phase1_model"],
        """
        These excerpts capture the first stable shape of the project. The data path is direct: scale one feature, build one target, reshape once, and send the result into a three-layer recurrent stack. The strength of that simplicity is that there is very little ambiguity about what the model is learning. The weakness is that almost every real-world forecasting complication is hidden outside the model rather than represented inside it.

        From a maintenance perspective, the snippets also show how tightly coupled the phase was. The script owns data acquisition, training, test preparation, plotting, and export in one place. That coupling is part of why later phases read like architectural corrections. The problem was never only that the model needed more power. The bigger problem was that experimentation got more expensive every time a new responsibility was added to the same file.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 2: TensorFlow Stacked LSTM Refinement", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        This refinement phase still sits in the December 2025 window. The staged `tensor.py` files were present in the initial repository history on December 18, 2025, and their local timestamps also point back to December 12, 2025. Because there is no later phase-specific commit trail for these files before the April 10, 2026 reorganization commit, the safest interpretation is that this stage belongs to the same early build-out period as the original baseline, but slightly later in conceptual sophistication.

        The dated significance of Phase 2 is therefore not a dramatic rewrite on one isolated day, but the fact that by late December 2025 the project had already moved beyond “one working script” into a repeatable experimentation pattern with better comments, clearer stacked-sequence intent, and TensorBoard visibility.
        """,
    )
    doc.add_paragraphs(
        """
        The second phase should be understood as consolidation rather than reinvention. The core forecasting pattern did not change dramatically, but the scripts became more deliberate about stacked recurrent layers, comments, and observability. In practical terms, this is where the project stopped behaving like a one-off experiment and started behaving like a repeated experiment. That shift matters, because once a workflow is repeated, visibility into training behavior becomes essential.

        TensorBoard support is a good example. It did not directly make the model more accurate, yet it changed how the developer interacted with the system. Histograms, graphs, and organized run logs introduced the idea that training needed instrumentation, not just terminal output. This kind of change often marks the difference between a script that can be demonstrated once and a script that can be improved over weeks or months.

        Conceptually, the phase also strengthened the developer's understanding of sequence shape. Stacked LSTMs require disciplined use of `return_sequences=True`, and a project that learns that lesson early becomes far easier to extend later. The importance of this stage is therefore educational and structural. It did not solve the big realism problem, but it taught the project how to reason about deeper recurrent compositions, and that knowledge became essential when the codebase started reaching for more expressive models.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 3: Dynamic Dropout and Better Training Control", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        Dynamic-dropout work is first traceable in the git history on December 18, 2025, including the commit `56c3a5a`, which explicitly mentions updates to `future-Eric`, a 30-day target, MAE loss, residual boosting, and lower dropout. Although `test_modified.py` itself carries the broader initial-import history, that dated commit is a strong signal that by December 18, 2025 the project had already started formalizing regularization and training-control ideas rather than leaving them static.

        In other words, Phase 3 belongs to the second half of December 2025, when the project began moving from a hardcoded TensorFlow prototype toward a script with policy-driven training behavior. That is the first point in the timeline where the code starts reacting to optimization dynamics across epochs instead of treating the entire run as a uniform process.
        """,
    )
    doc.add_paragraphs(
        """
        Phase 3 introduced one of the first genuinely strategic improvements in training behavior: dropout became dynamic rather than fixed. That may sound like a small tuning preference, but it reflects a deeper recognition that the needs of early training and late training are not the same. Heavy regularization can be useful when the network is first learning broad structure, yet the same amount of stochastic disruption can become counterproductive once the model is trying to sharpen details.

        By decaying dropout over time, the project effectively turned regularization into a schedule instead of a constant. The callback-based implementation is notable because it also marks a shift in how the team related to the training loop. Rather than accepting the default behavior of the framework, the script began to inject policy into the process. That pattern becomes even more important later, when schedulers, checkpointing, and Monte Carlo inference start to shape not just optimization but the actual behavior of the forecasting output.

        This phase also improved operator experience. TensorBoard logging, explicit epoch counts, and readable callback output made the script easier to monitor during longer runs. Those kinds of improvements are often undervalued in short notes, but they matter in real practice. A model that is slightly better in theory but difficult to inspect is often less useful than a model that is somewhat simpler but clearly observable. The project was starting to learn that productive experimentation depends on feedback quality as much as model complexity.

        At the same time, dynamic dropout did not remove the core structural constraints of the TensorFlow era. The code was still mostly single-feature, still monolithic, and still tied to a framework whose abstractions became uncomfortable once the project wanted more custom control. That is why this phase reads as a serious upgrade but not yet a destination. It solved an optimization problem while leaving the broader architecture problem open.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 3: Fixed Dropout Becomes Scheduled Dropout",
        "Earlier Static Recurrent Stack",
        snippets["phase1_model"],
        "Dynamic Dropout Callback and Rebuilt Stack",
        snippets["phase3_dynamic"],
        """
        The comparison shows a clear philosophical change. The earlier model definition treats dropout as a property of the architecture: once chosen, it stays the same for the entire run. The later implementation treats dropout as part of the training policy. That distinction matters because it acknowledges that regularization is temporal, not merely structural.

        It is also one of the first places where the project moved from assembling layers to actively steering them. The callback reaches into the model at epoch boundaries, updates multiple layers in place, and keeps the operator informed about the current regularization strength. In other words, the project began to act less like a consumer of framework defaults and more like a system that defined its own operating rhythm.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 4: Extended TensorFlow Forecasting", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        This phase is one of the clearest in the repository timeline because `future-Eric.py` has a dense run of dated commits on January 2, 2026 and January 4, 2026. On January 2, 2026, commits such as `7451794` and `adfa880` show the model being modified for multi-step prediction and early stopping. On January 4, 2026, a burst of changes including `a9ac210`, `61e4c16`, `d28e8f5`, and `e8fc461` reflects intense iteration around disappearing predicted days, sigmoid behavior, and general forecasting instability.

        That cluster of exact dates matters because it shows this phase was not a vague gradual drift. It was a concentrated early-January 2026 expansion in which richer feature engineering, deeper callback discipline, and future-horizon logic were all being actively hammered into shape under obvious engineering pressure.
        """,
    )
    doc.add_paragraphs(
        """
        Phase 4 is the point where the project decisively outgrew close-only forecasting. The code began engineering returns, moving averages, RSI, calendar context, and a wider set of technical indicators. This is not just a larger feature matrix. It is a statement that the model should be given structured hints about the market rather than being asked to infer every useful pattern from raw closing prices alone. In practical forecasting work, that often marks the difference between a model that broadly follows trend and a model that can respond to localized regime changes.

        The feature expansion forced improvements elsewhere. Once the model had separate target behavior and non-target feature behavior, scaling had to become more nuanced. A single scaler was no longer enough. The code therefore introduced separate transformations for returns and auxiliary features, and then merged the scaled outputs into a coherent training matrix. This is an engineering sign of maturity: preprocessing is no longer shaped around convenience, but around the actual statistical roles played by different variables.

        Training control also became more disciplined in this phase. `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` changed the script from a linear train-and-hope workflow into a monitored optimization process with explicit recovery and selection behavior. That matters even more in finance-adjacent sequence work, where a slightly overtrained model can look impressive on one run and then fail to generalize on the next.

        The most ambitious idea in the phase is residual correction. LSTMs tend to smooth predictions because minimizing aggregate loss encourages average behavior. The project responded by training a secondary regressor on the residuals of the LSTM output. That is a creative hybrid pattern: let the recurrent network learn broad temporal structure, then let a lighter model correct the places where temporal smoothing misses sharp local moves. Whether or not every experimental variant was perfect, the phase shows a clear willingness to compose multiple tools instead of expecting one network to do all the work alone.

        Importantly, the same phase also reveals the growing pain that made TensorFlow harder to justify. The logic for indicators, callbacks, residual correction, validation, and future forecasting was accumulating inside increasingly heavy scripts. The machine-learning ideas were advancing, but the framework and file structure were becoming less comfortable under the project's experimental load. That tension directly sets up the PyTorch migration.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 4: From Close-Only Inputs to Multi-Feature Engineering",
        "Leakage-Prone Single-Scaler Test Handling",
        snippets["phase1_leakage"],
        "Extended TensorFlow Multi-Feature Preparation",
        snippets["phase4_features"],
        """
        The short TensorFlow snippet on the left captures an early weakness that became much more visible once the project moved into richer forecasting tasks. Re-fitting the scaler around model inputs that overlap the test horizon may be expedient, but it weakens the cleanliness of the evaluation setup. That became harder to justify once the pipeline started claiming multi-step forecasting improvements.

        The January 2026 feature-preparation excerpt shows the corrective direction. A richer feature space requires more careful handling of missing values, indicator failures, and scaling boundaries, and it also forces the code to admit that the target series and the supporting features do not necessarily belong under one statistical transformation. That is why this comparison belongs in Phase 4: preprocessing became a first-class subsystem at exactly the point where the project tried to make the forecast more realistic.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 5: Validation Discipline and Residual Correction",
        "TensorFlow Phase 4 Training Controls",
        snippets["phase4_residual"],
        "Later Stable PyTorch Training Controls",
        snippets["phase6_training"],
        """
        The TensorFlow excerpt demonstrates the project's first serious attempt to formalize training governance. Instead of assuming the final epoch is best, the script begins watching validation loss, reducing the learning rate when progress stalls, and preserving the best checkpoint. That is a major change in scientific discipline even before the framework migration occurs.

        The later PyTorch excerpt shows how those same instincts continued but in a more explicit training-loop environment. The core philosophy stayed the same: volatile sequence forecasting needs guardrails. What changed was the degree of control the developer had over those guardrails and the ease with which they could be composed.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 5: PyTorch Transition", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        The transition to PyTorch is anchored by `future-eric-pytorch.py`, whose surviving file history points to January 21, 2026 through commit `66b0a20`, with local file timestamps landing on January 26, 2026. That places the framework migration squarely in late January 2026, after the early-January TensorFlow expansion had already exposed the need for more explicit control.

        This timing is important because it shows the migration was not part of the original design. It was a response to the difficulty of extending the TensorFlow-based code under growing forecasting and hardware demands. By late January 2026, the repository had reached the point where a framework rewrite was cheaper than continuing to force every new requirement into the older structure.
        """,
    )
    doc.add_paragraphs(
        """
        The shift to PyTorch was driven by more than preference. The repository documentation makes it clear that hardware behavior on AMD/ROCm, difficulty with custom forecasting control, and general framework friction were becoming real blockers. PyTorch offered a different contract with the developer: fewer opaque conveniences, but much clearer control over tensors, devices, modules, and loops. For a project that was increasingly experimental, that trade-off was attractive.

        One of the earliest benefits of the move is visible in device routing. Tensors are explicitly pushed onto the chosen device, and the model class becomes a normal Python object derived from `nn.Module` rather than an object assembled exclusively through high-level Keras stacking. That shift can feel more verbose, but it pays back immediately in debuggability. Shape assumptions, intermediate states, and forward-pass logic become easier to inspect and reason about.

        This phase is also where the repository begins demonstrating more intentional control over architecture search and persistence. Optuna support, SQLite-backed study storage, and more systematic parameterization show a project moving away from guesswork. Hyperparameter exploration is no longer only a mental activity carried out by manually editing values. It becomes something the code can record, resume, and compare across runs.

        The PyTorch transition was not instantly clean. Like many migrations, it initially carried some conceptual baggage from the older framework. But even in that transitional form, the rewrite solved an important problem: it gave the project room to keep evolving. Once the code could directly express device behavior and model internals, the next stages of stabilization and full-pipeline orchestration became much easier to justify.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 6: Keras Style Gives Way to Explicit PyTorch Control",
        "Device-Aware Tensor Placement",
        snippets["phase5_device"],
        "Explicit PyTorch Module Definition",
        snippets["phase5_model"],
        """
        The device-routing excerpt is deceptively small, but it signals a major operational change. The project is no longer trusting the framework to decide how tensors should move. It is explicitly choosing the execution context and making that choice part of the visible code path. That makes hardware behavior easier to reason about and easier to debug.

        The class definition completes the picture. In Keras, the model architecture was primarily described by stacking layers into a container. In PyTorch, the project now owns the `forward()` method and the internal composition logic. That opens the door to custom behavior, experimental layer choreography, and easier adaptation when the model stops fitting the assumptions of a higher-level API.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 6: PyTorch Stable Core", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        The stable-core phase is visible in the March 28 to March 29, 2026 history of `pytorch_fixed.py`. Commit `b035389` on March 28, 2026 explicitly describes bug fixes and training improvements, including the removal of a deprecated verbose argument, a higher epoch count, and more patient early stopping. The next day, March 29, 2026, commits `5c7f01d` and `43f741e` show continued work around the updated LSTM model and integration into the wider application flow.

        Those dates make this phase easy to interpret: late March 2026 was when the repository stopped merely having a PyTorch version and started hardening that version into a dependable baseline. Stability improvements, not just feature additions, became the focus.
        """,
    )
    doc.add_paragraphs(
        """
        Once the migration shock settled, the repository entered a stabilization phase. `pytorch_fixed.py` is significant because it behaves like a baseline the rest of the PyTorch work can trust. The model is not trying to do everything at once. Instead, it organizes data into datasets and loaders, standardizes technical indicators, adopts bidirectional recurrent layers, and formalizes validation-aware training decisions. This is the phase where the project starts to look less like a port and more like a coherent PyTorch-native implementation.

        The choice of `HuberLoss` is especially telling. Financial time series produce spikes, abrupt moves, and outliers that can dominate ordinary MSE training. By switching to a loss that behaves quadratically near the origin and linearly farther out, the project made an explicit statement about robustness. It is no longer enough for the model to minimize average squared error. The model must also remain sane in the presence of irregular market shocks.

        Gradient clipping and learning-rate scheduling reinforce the same theme. Recurrent models can fail dramatically when gradients explode, and long training runs can stagnate if the optimizer never transitions into a finer search regime. By clipping norms and using `ReduceLROnPlateau`, the code defends both against catastrophic instability and against wasteful persistence at an overly aggressive step size. These are the kinds of changes that make a model feel less fragile in day-to-day use.

        The stable core phase also matters because it clarifies priorities. Accuracy, stability, and interpretability of training behavior are treated as equally important. That balance is what allows the next phase to become larger and more feature-rich without collapsing immediately under its own complexity.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 7: A Stable PyTorch Baseline",
        "Bidirectional Model Definition",
        snippets["phase6_model"],
        "Robust Loss, Scheduler, and Clipping",
        snippets["phase6_training"],
        """
        The model definition shows how the project moved beyond a straight port of the old recurrent stack. Bidirectionality increases representational capacity without changing the forecasting objective, and the modular layer construction makes it easier to vary architecture depth. The code is more explicit, but it is also more legible about what information is flowing where.

        The training-control excerpt matters just as much as the architecture. In a volatile sequence domain, the most useful model is often the one that degrades gracefully rather than the one that occasionally produces the absolute best run. Huber loss, scheduler logic, and gradient clipping are all mechanisms for graceful degradation.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 7: Full PyTorch Pipeline", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        The full-pipeline phase has the richest dated trail in the entire repository. The foundation appears on February 22, 2026 with commits `51c481e` and `5bf22a7`, which add Optuna support and a data dictionary. It accelerates on March 16, 2026 (`b9711de`) with bidirectional multi-feature PyTorch prediction, on March 18, 2026 (`26842fb`) with MACD, Bollinger Bands, Monte Carlo dropout, and logging, and on March 24 to March 29, 2026 with interactive ticker selection, detailed plotting, GUI-related integration, and conversation logging. The latest concentrated refinements arrive on April 7 and April 9, 2026 through commits `661deff`, `eb17333`, and `4604708`, which improve training configuration, batch processing, API handling, and future prediction behavior.

        Because these dates are so dense, this phase should be read as a sustained build cycle spanning late February through early April 2026. It was not one feature drop. It was the period when the project assembled nearly every major PyTorch-era capability into one large operating script.
        """,
    )
    doc.add_paragraphs(
        """
        `pytorch_plotted.py` represents the apex of the monolithic era. It is the stage where the repository combines hardware workarounds, rich feature engineering, optimizer flexibility, logging, uncertainty estimation, and polished plotting into one large script. Seen one way, this file is a triumph: it centralizes an enormous amount of learned project knowledge. Seen another way, it is a warning sign, because success came bundled with increasing difficulty of maintenance.

        The hardware management in this phase is especially mature. Instead of assuming all CUDA-like environments behave the same way, the code recognizes ROCm-specific realities and selectively disables or enables backend behavior accordingly. That means the runtime configuration itself becomes conditional engineering logic. The project is no longer merely running on hardware; it is negotiating with hardware and driver ecosystems that do not behave identically.

        Mixed precision and optimizer abstraction also show a move toward throughput-aware design. The code is no longer asking only how to train a model. It is asking how to train it efficiently, how to keep console progress readable over long runs, and how to make optimizer experimentation part of the workflow instead of a disruptive manual rewrite. These are concerns that appear when a project has survived long enough to outgrow its original experimental tempo.

        The most memorable achievement of this phase is the solution to the jagged trajectory problem. Deterministic or mean-aggregated forecasts looked implausibly smooth, which is visually and conceptually wrong for a market forecast. By keeping dropout active at inference time and using a single stochastic realization for the displayed path while reserving the full run distribution for interval width, the script finally aligned its future chart behavior with the irregularity users expected to see.

        Yet the very richness of the script created the next architectural pressure. Once plotting, artifact management, training, device setup, uncertainty estimation, and data preparation all lived in the same file, improving one area meant navigating all the others. The code was powerful, but power was becoming expensive to edit. That is the exact situation that typically triggers a separation of concerns rewrite, which is what happened next.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 8: Hardware-Aware Runtime Configuration",
        "ROCm-Aware Backend Controls",
        snippets["phase7_backend"],
        "AMP and Optimizer Flow",
        snippets["phase7_amp"],
        """
        These excerpts demonstrate how deeply operational knowledge had entered the codebase by this point. Backend flags, TF32 settings, environment variables, optimizer indirection, and AMP logic all live alongside the model itself. That concentration of detail is one reason the script was so capable. It is also one reason it became difficult to treat as a simple forecasting file.

        The broader significance is that performance and correctness are now intertwined. The runtime path is not just an implementation detail. It changes whether the experiment runs reliably on the intended workstation and whether long sessions remain practical.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 9: Monte Carlo Dropout for Realistic Future Paths",
        "Deterministic Evaluation Would Smooth the Forecast",
        snippets["phase6_training"],
        "Stochastic Inference for Jagged Trajectories",
        snippets["phase7_mc"],
        """
        The left-hand excerpt is not wrong; it simply belongs to a different concern. It shows a stable supervised training loop, which is necessary but insufficient for believable multi-step future charts. The right-hand excerpt solves a specifically visual and probabilistic problem: how to display a future path that looks like a possible market trajectory rather than an averaged curve.

        That distinction is central to the maturity of the project. Later phases are not only about better loss curves. They are about better outputs for the human who has to interpret the forecast. The confidence band and the chosen visible path serve different purposes, and this phase makes that separation explicit.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 8: PyTorch Experiments", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        The experimental branch overlaps heavily with the full-pipeline build, but its dated trail still helps. `pytorch_additional.py` inherits the same February 22, 2026 Optuna-era base, then picks up major momentum on March 16, March 18, March 24, and March 27, 2026 as the repository explores richer multi-feature forecasting, Monte Carlo dropout, interactive selection, and more expressive architectural ideas. Its local creation timestamp on March 24, 2026 also matches the point where experimentation became a visible parallel track rather than just ad hoc edits inside the main plotted script.

        In practical terms, this means the experimental phase belongs to late March 2026, when the mainline PyTorch work was already strong enough that the repository could afford to test attention and residual ideas without losing the stable forecasting path.
        """,
    )
    doc.add_paragraphs(
        """
        The experimental phase extends the full pipeline rather than replacing it. Once the main script was working, the natural next question became whether the architecture could be made more expressive or more efficient. That is why this stage contains attention layers, residual ideas, deeper fully connected heads, cyclical features, and broader performance experimentation. The repository was no longer searching only for basic correctness. It was testing the boundaries of how much complexity the current design could support.

        The attention-enhanced experiments are important because they show the project reasoning about sequence representation at a higher level. A plain LSTM can capture temporal dependencies, but attention offers a different mechanism for highlighting which positions in the sequence deserve emphasis. Adding that mechanism suggests the project had reached a point where “another recurrent layer” was no longer the only conceptual tool under consideration.

        Just as importantly, this phase sharpened the argument for architectural separation. When advanced attention blocks, optimizer experiments, and heavy plotting all share one operational space, the maintenance cost rises quickly. In that sense, the experimental files are both valuable research artifacts and evidence that the monolithic layout had hit its practical limit.
        """
    )

    doc.add_heading("Code Exhibit 10: Attention and Residual Experiments", level=2)
    doc.add_code_block(snippets["phase8_attention"])
    doc.add_paragraphs(
        """
        This excerpt shows how ambitious the experimental branch became. The model is no longer a straightforward recurrent stack with a single dense output. It contains attention, normalization, residual mixing, and a cascading fully connected head designed for multi-step output. Whether or not this specific design became the canonical future, it demonstrates the level of architectural curiosity the project had reached.
        """
    )

    doc.add_page_break()
    doc.add_heading("Phase 9: Separated `cpp` Pipeline", level=1)
    add_timeline(
        doc,
        "Date Range and Milestones",
        """
        The separated pipeline is cleanly dated. Both `pytorch_train_cpp.py` and `pytorch_plot_cpp.py` were created on April 9, 2026, and then reorganized into the current layout by commit `f9efa69` on April 10, 2026. The preceding April 9, 2026 commit `4604708` also shows that the project was still modifying future-prediction behavior in the monolithic flow immediately before the split.

        Those dates make the architectural pivot very concrete. The separation happened at the end of the observed timeline, in direct response to the complexity accumulated through March and early April 2026. By April 10, 2026, the repository had explicitly chosen stable outputs, reusable artifacts, and train-versus-plot decoupling as the next stage of maturity.
        """,
    )
    doc.add_paragraphs(
        """
        The separated `cpp`-style pipeline is arguably the most important architectural milestone in the entire project. It does not introduce the flashiest neural-network idea, but it solves one of the most persistent engineering problems: the all-in-one script had become too expensive to change. Training and plotting were deeply entangled, so even cosmetic or reporting-focused changes could drag the full inference stack into the loop.

        Splitting the workflow into `pytorch_train_cpp.py` and `pytorch_plot_cpp.py` changed the ergonomics of the entire project. The training script became a compute-focused artifact producer. Its job is to fetch data, engineer features, train or load the model, generate predictions and future rollouts, and then save those outputs into an organized directory structure. The plotting script became an artifact consumer. Its job is to discover those saved outputs and render them without repeating the expensive computation that produced them.

        That separation improved reliability, reproducibility, and development speed at the same time. Reliability improved because plotting failures no longer implied model-training failures. Reproducibility improved because the pipeline now has named outputs in stable directories rather than ad hoc files scattered according to the active working directory. Development speed improved because charting changes can be tested by reusing existing CSV artifacts instead of rerunning long training sessions.

        The new training script also shows attention to computational efficiency. Vectorized window generation via `sliding_window_view` reduces Python-loop overhead, MultiIndex flattening hardens the pipeline against `yfinance` quirks, and optional `torch.compile` reflects a project now interested in throughput as a formal design concern. These are not cosmetic refinements. They are the signs of a script that expects to be rerun often and therefore cares about the cumulative cost of small inefficiencies.

        On the plotting side, auto-discovery of the latest predictions and future CSVs is a surprisingly meaningful usability improvement. The script meets the operator halfway: if the expected artifacts exist, it finds them and continues. This kind of convenience is exactly what mature tooling should provide. It reduces friction without hiding where the data came from.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 11: Organized Training Artifacts and Vectorized Windows",
        "Separated Output Layout",
        snippets["phase9_paths"],
        "Vectorized Sequence Generation",
        snippets["phase9_vector"],
        """
        The training-side excerpts make the architectural intent explicit. Outputs are no longer an afterthought, and sequence creation is no longer a purely Python-loop exercise unless the optimized path fails. Both changes reflect the same underlying idea: the pipeline should be fast, predictable, and easy to audit.

        It is worth noting that these changes are not only about speed. Stable output folders and deterministic naming conventions are also documentation. They tell a future reader where to look for models, prediction backtests, and future rollouts without requiring them to reverse-engineer the script's runtime choices.
        """
    )

    add_comparison(
        doc,
        "Code Exhibit 12: Optional Compilation and Plot Artifact Discovery",
        "Training-Time Compile Hook",
        snippets["phase9_compile"],
        "Plot-Side Auto-Discovery",
        snippets["phase9_plot_discovery"],
        """
        The compile hook on the training side and the discovery logic on the plotting side together summarize the maturity of the separated architecture. Training is optimized when the environment allows it, but the code degrades gracefully if compilation is skipped. Plotting similarly optimizes for convenience without becoming brittle: it searches several sensible locations and selects the newest matching artifact.

        This is the strongest current expression of the project's accumulated lessons. Performance improvements are treated as optional accelerators, not assumptions. Operational quality-of-life improvements are treated as real features, not disposable polish.
        """
    )

    doc.add_page_break()
    doc.add_heading("Cross-Cutting Design Themes", level=1)
    doc.add_paragraphs(
        """
        The first recurring theme is the gradual externalization of structure. Early code keeps almost everything implicit inside one script. Later code begins naming subsystems: feature engineering, validation policy, artifact roots, forecast uncertainty, plotting discovery, and optimizer configuration. Naming matters because it lets the repository hold more complexity without becoming unreadable.

        The second theme is that realism repeatedly challenged convenience. Close-only forecasting was convenient, but not rich enough. Mean future trajectories were convenient, but not believable. Monolithic plotting was convenient, but too expensive to iterate on. The project's history can therefore be read as a sequence of moments where convenient defaults were abandoned in favor of behavior that better matched the problem domain.

        The third theme is hardware humility. The repository did not assume that accelerator support would simply behave. It had to learn what ROCm tolerated, what needed to be disabled, where mixed precision helped, and where it might not. This matters because the project is not just a mathematical object. It is a system that lives on a specific workstation and inherits the reality of that machine.

        The fourth theme is that maintainability became inseparable from model quality. A forecasting script that cannot be rerun cleanly, logged safely, or visualized without retraining is harder to trust and harder to improve. By the end of the timeline, software-architecture decisions are not ancillary to the forecasting goal. They are part of what makes the forecasting goal achievable.
        """
    )

    doc.add_page_break()
    doc.add_heading("Error and Issue Analysis", level=1)
    doc.add_paragraphs(
        """
        One persistent issue across the project was backend fragility on AMD-oriented setups, especially when LSTM-heavy workloads met ROCm or MIOpen edge cases. The repository's response was not to pretend the hardware problem did not exist, but to encode defensive behavior directly into the runtime path. That is a mature move. A system becomes more reliable not only by using better algorithms, but by being honest about the environment it must survive in.

        Another recurring issue was output smoothness. In many time-series projects, a forecast that looks smooth can be mistaken for a forecast that is strong. Here, the opposite lesson emerged. Over-smoothed trajectories were understood as a modeling and presentation failure because they did not resemble the stochastic behavior users expected from financial series. The Monte Carlo dropout solution therefore corrected both the statistical representation and the human interpretability of the result.

        Data-shape inconsistencies, especially from `yfinance`, formed a quieter but equally important category of bugs. MultiIndex columns, rolling-window NaNs, and scaling assumptions are the sort of issues that do not always crash immediately. They can instead poison downstream tensors or silently distort evaluation. The repository's later emphasis on column flattening, `ffill`/`bfill`, and clearer preprocessing stages reflects a hard-earned awareness that data plumbing bugs are often more dangerous than obvious syntax errors.

        Experiment-management drift was another problem that the project solved gradually. Optuna databases, prediction CSVs, future forecasts, and plot outputs all have long-term value only if they can be found again. Earlier phases tolerated looser working-directory assumptions. Later phases made output roots script-relative and artifact categories explicit. That is not merely tidiness. It is how the project preserved the ability to compare runs over time.

        User experience issues also mattered. Broken progress bars, noisy print statements, and mixed console output can turn long training runs into frustrating sessions. The move toward `tqdm`-safe output and cleaner logging is easy to dismiss as cosmetic, but in practice it improves trust. A tool that communicates clearly during long operations is easier to leave running, easier to monitor, and easier to debug when something goes wrong.

        Taken together, these issues show that the hardest problems were rarely isolated to one layer of the stack. Forecasting realism, runtime stability, and maintainability kept interacting. The project's success came from treating those interactions as real engineering work instead of as distractions from the “main” ML task.
        """
    )

    doc.add_page_break()
    doc.add_heading("Canonical End State", level=1)
    doc.add_paragraphs(
        """
        If a future contributor wants the clearest view of the current architecture, the separated scripts are the best starting point: `REWRITE/separated/pytorch_train_cpp.py` and `REWRITE/separated/pytorch_plot_cpp.py`. Together they preserve the lessons of the monolithic PyTorch pipeline while removing the coupling that made experimentation painful. They are not the beginning of the story, but they are the cleanest synthesis of it.

        The most important bridge file before that separation remains `REWRITE/learning-path/07-pytorch-full-pipeline/pytorch_plotted.py`. Anyone trying to understand how the mature feature set was assembled should read that file as the concentrated form of the project's mid-to-late intelligence. Anyone trying to understand how that intelligence was made sustainable should then read the separated scripts immediately afterward.

        In other words, the present architecture did not appear because the project suddenly discovered a better model. It appeared because repeated contact with real constraints taught the codebase how to organize itself. The final lesson of the repository is therefore larger than forecasting: it is an example of iterative engineering pressure gradually producing a more durable system.
        """
    )

    return doc


def main() -> None:
    docs_dir = Path(__file__).resolve().parent
    output_path = docs_dir / "PROJECT_LOG_FULL_EXPANDED_REPORT.docx"
    builder = build_report()
    final_path = output_path
    try:
        write_docx(output_path, builder)
    except PermissionError:
        fallback = docs_dir / f"PROJECT_LOG_FULL_EXPANDED_REPORT_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        write_docx(fallback, builder)
        final_path = fallback
    print(f"Wrote: {final_path}")
    print(f"Body words: {builder.body_word_count}")
    print(f"Code lines: {builder.code_line_count}")
    print(f"Estimated pages: {builder.estimate_pages()}")


if __name__ == "__main__":
    main()
