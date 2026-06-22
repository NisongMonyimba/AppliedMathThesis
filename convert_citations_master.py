#!/usr/bin/env python3
"""
Master citation-conversion script. Converts every prose-style citation
("(Author, Year)" / "Author (Year)" / "Author et al., Year", including
instances broken across line wraps) throughout the thesis chapter files
into proper natbib \citep{}/\citet{} commands linked to references.bib.

Run from the manuscript/ directory:
    python3 convert_citations_master.py chapters
"""
import re
import sys
from pathlib import Path

# Author-name regex (using \s+ so line-wraps don't break matching) -> bib key
AUTHORS = [
    (r"Huang,?\s+Malham\\'e,?\s+and\s+Caines", "huangmalhamecaines2006"),
    (r"Carmona\s+and\s+Delarue", "carmonadelarue2018"),
    (r"Carmona--Delarue", "carmonadelarue2018"),
    (r"Carmona\s+\\&\s+Delarue", "carmonadelarue2018"),
    (r"Lasry\s+and\s+Lions", "lasrylions2007"),
    (r"Cardaliaguet,?\s+Delarue,?\s+Lasry,?\s+and\s+Lions", "cardaliaguetetal2019"),
    (r"Cardaliaguet\s+et\s+al\.?", "cardaliaguetetal2019"),
    (r"Ahuja,?\s+Ren,?\s+and\s+Yang", "ahujarenyang2019"),
    (r"Ahuja\s+et\s+al\.?", "ahujarenyang2019"),
    (r"Cardaliaguet,?\s+Seeger,?\s+and\s+Souganidis", "cardaliaguetseegersouganidis2025"),
    (r"Cardaliaguet\s+and\s+Lehalle", "cardaliaguetlehalle2018"),
    (r"Cardaliaguet--Lehalle", "cardaliaguetlehalle2018"),
    (r"Lehalle\s+and\s+Mouzouni", "lehallemouzouni2019"),
    (r"Lehalle--Mouzouni", "lehallemouzouni2019"),
    (r"Neuman\s+and\s+Vo\{\\ss\}", "neumanvoss2023"),
    (r"Angiuli,?\s+Fouque,?\s+and\s+Lauri\\`ere", "angiulietal2022"),
    (r"Angiuli,?\s+Fouque,?\s+(?:and\s+)?Lauriere", "angiulietal2022"),
    (r"Angiuli,?\s+Fouque,?\s+Lauri\\`ere,?\s+and\s+Zhang", "angiulietal2024"),
    (r"Angiuli\s+et\s+al\.?", "angiulietal2024"),
    (r"Hu\s+and\s+Lauri\\`ere", "hulauriere2023"),
    (r"Ma,?\s+Li,?\s+and\s+Zhang", "malizhang2024"),
    (r"Ma--Li--Zhang", "malizhang2024"),
    (r"Fama\s+and\s+French", "famafrench1993"),
    (r"Khandani\s+and\s+Lo", "khandanilo2011"),
    (r"Almgren\s+and\s+Chriss", "almgrenchriss2000"),
    (r"Carmona,?\s+Fouque,?\s+and\s+Sun", "carmonafouquesun2015"),
    (r"Carmona--Fouque--Sun", "carmonafouquesun2015"),
    (r"Casgrain\s+and\s+Jaimungal", "casgrainjaimungal2020"),
    (r"Fournier\s+and\s+Guillin", "fournierguillin2015"),
    (r"Fournier--Guillin", "fournierguillin2015"),
    (r"Esfahani\s+and\s+Kuhn", "esfahanikuhn2018"),
    (r"Blanchet,?\s+Kang,?\s+and\s+Murthy", "blanchetkangmurthy2019"),
    (r"Kim,?\s+Korajczyk,?\s+and\s+Neuhierl", "kimkorajczykneuhierl2015"),
    (r"Villani", "villani2009"),
    (r"Karatzas\s+and\s+Shreve", "karatzasshreve1991"),
    (r"Karatzas--Shreve", "karatzasshreve1991"),
    (r"Kloeden\s+and\s+Platen", "kloedenplaten1992"),
    (r"El\s+Karoui\s+et\s+al\.?", "elkarouipengquenez1997"),
    (r"El\s+Karoui,?\s+Peng,?\s+and\s+Quenez", "elkarouipengquenez1997"),
    (r"Chassagneux,?\s+Crisan,?\s+and\s+Delarue", "chassagneuxcrisandelarue2019"),
    (r"Chassagneux--Crisan--Delarue", "chassagneuxcrisandelarue2019"),
    (r"Achdou\s+and\s+Porretta", "achdouporretta2016"),
    (r"Achdou--Porretta", "achdouporretta2016"),
    (r"Lord,?\s+Powell,?\s+and\s+Shardlow", "lordpowellshardlow2014"),
    (r"Bencheikh", "bencheikh2020"),
    (r"Gobet,?\s+Lemor,?\s+and\s+Warin", "gobetlemorwarin2005"),
    (r"Bouchard\s+and\s+Touzi", "bouchardtouzi2004"),
    (r"Cuturi\s+and\s+Doucet", "cuturidoucet2014"),
    (r"Cuturi--Doucet", "cuturidoucet2014"),
    (r"Cuturi", "cuturi2013"),
    (r"Agueh\s+and\s+Carlier", "aguehcarlier2011"),
    (r"Agueh--Carlier", "aguehcarlier2011"),
    (r"Borkar's", r"BORKARPOSSESSIVE"),  # handled specially below
    (r"Borkar,?\s+1997,?\s+2008", "borkar1997,borkar2008"),
    (r"Borkar", "borkar2008"),
    (r"Perrin\s+et\s+al\.?", "perrinetal2022"),
    (r"Zhou,?\s+Zhou,?\s+and\s+Hu", "zhouzhouhu2025"),
    (r"Haarnoja\s+et\s+al\.?", "haarnojaetal2018"),
    (r"Schulman\s+et\s+al\.?", "schulmanetal2017"),
    (r"Fujimoto\s+et\s+al\.?", "fujimotoetal2018"),
    (r"Zaheer\s+et\s+al\.?", "zaheeretal2017"),
    (r"Sutton\s+and\s+Barto", "suttonbarto2018"),
    (r"Carmona,?\s+Delarue,?\s+and\s+Lachapelle", "carmonadelaruelachapelle2014"),
    (r"Zeng\s+et\s+al\.?", "zengetal2024"),
    (r"Garnier,?\s+Papanicolaou,?\s+and\s+Yang", "garnierpapanicolaouyang2013"),
    (r"Firoozi\s+et\s+al\.?", "firoozietal2019"),
    (r"Ren\s+and\s+Firoozi", "renfiroozi2024"),
    (r"Cuchiero,?\s+Reisinger,?\s+and\s+Rigger", "cuchieroreisingerrigger2024"),
    (r"Kydland\s+and\s+Prescott", "kydlandprescott1977"),
    (r"Givens\s+and\s+Shortt", "givensshortt1984"),
    (r"Olkin\s+and\s+Pukelsheim", "olkinpukelsheim1982"),
    (r"Chen,?\s+Hong,?\s+and\s+Stein", "chenhongstein2002"),
    (r"Frazzini,?\s+Israel,?\s+and\s+Moskowitz", "frazziniisraelmoskowitz2018"),
    (r"Andrews", "andrews1991"),
    (r"Kyle", "kyle1985"),
    (r"Jaimungal\s+and\s+Nourian", "jaimungalnourian2015"),
    (r"Jaimungal--Nourian", "jaimungalnourian2015"),
    (r"Carmona\s+and\s+Wang", "carmonawang2021"),
    (r"Precup,?\s+Sutton,?\s+and\s+Singh", "precupsuttonsingh2000"),
    (r"Lange,?\s+Gabel,?\s+and\s+Riedmiller", "langegabelriedmiller2012"),
    (r"Zhang", "zhang2017"),
    (r"Billingsley", "billingsley1999"),
]

YEAR_SUFFIX = re.compile(r",?\s+\(?20\d\d[a-z]?\)?")


def is_already_cited(text: str, start: int) -> bool:
    preceding = text[max(0, start - 15):start]
    return "citep{" in preceding or "citet{" in preceding


def convert_text(text: str) -> tuple[str, int]:
    n_subs = 0
    for author_re, key in AUTHORS:
        if key == "BORKARPOSSESSIVE":
            pattern = re.compile(author_re)
            result, count = [], 0
            last_end = 0
            for m in pattern.finditer(text):
                if is_already_cited(text, m.start()):
                    continue
                result.append(text[last_end:m.start()])
                result.append(r"\citet{borkar1997,borkar2008}'s")
                last_end = m.end()
                count += 1
            result.append(text[last_end:])
            text = "".join(result)
            n_subs += count
            continue

        pattern = re.compile(author_re + YEAR_SUFFIX.pattern)
        result, count = [], 0
        last_end = 0
        for m in pattern.finditer(text):
            if is_already_cited(text, m.start()):
                continue
            result.append(text[last_end:m.start()])
            if "," in key:
                result.append(r"\citep{" + key + "}")
            else:
                result.append(r"\citep{" + key + "}")
            last_end = m.end()
            count += 1
        result.append(text[last_end:])
        text = "".join(result)
        n_subs += count

    return text, n_subs


def manual_fixups(text: str, fname: str) -> str:
    """Hand-verified fixups for known awkward/redundant phrasing left after
    the automated passes (cross-checked against a full compile)."""
    if fname == "02_mathematical_preliminaries.tex":
        text = text.replace(
            r"we recommend \citep{villani2009} for optimal transport, \citep{carmonadelarue2018}, Vol.\ I, Chapters 1--2 for Wasserstein analysis in the MFG context, and\n\citep{karatzasshreve1991} for stochastic calculus.",
            r"we recommend \citet{villani2009} for optimal transport,\n\citet[Vol.~I, Chapters 1--2]{carmonadelarue2018} for Wasserstein analysis in the MFG context, and\n\citet{karatzasshreve1991} for stochastic calculus."
        )
        text = re.sub(
            r"\\citep\{carmonadelarue2018\},?\s*Vol\.\\?\s*I,\s*Chapter\s*5\)?\s*for a comprehensive treatment",
            lambda m: r"\citet[Vol.\ I, Chapter 5]{carmonadelarue2018} for a comprehensive treatment",
            text
        )
    if fname == "01_introduction.tex":
        text = text.replace("Theorem~B.9", r"Theorem~\ref{thm:wellposedness}")
    if fname == "04_numerical_analysis.tex":
        text = re.sub(
            r"The CEMRACS\s*\n?\s*2017 project \(see \\citep\{cemracs2017\}\)",
            lambda m: r"The \citep{cemracs2017} project",
            text
        )
    if fname == "10_synthesis_outlook.tex":
        text = text.replace(
            "A 2024 paper in the Journal of Optimization Theory\nand Applications, ``Incomplete Information Mean-Field Games and Related Riccati\nEquations'', studies",
            r"\citet{incompleteinfomfg2024} study"
        )
        text = re.sub(r"\(JOTA,?\s*2024\)", r"\\citep{incompleteinfomfg2024}", text)
        text = text.replace(
            "Incomplete Information MFG \\citep{incompleteinfomfg2024}; Zeng et al.\\",
            r"\citep{incompleteinfomfg2024,zengetal2024} &"
        )
        text = text.replace(
            "``Learning Macroeconomic Policies through Dynamic\nStackelberg Mean-Field Games'' (2025)",
            r"\citep{macromfg2025}"
        )
        text = text.replace(
            "``Heterogenous Macro-Finance Model: A Mean-field Game Approach'' (2025)",
            r"\citep{heteromacrofinance2025}"
        )
        text = re.sub(r"Macro MFG \(arXiv,?\s*2025\)", r"\\citep{macromfg2025}", text)
        text = re.sub(r"\(COLT\s*2022\)", r"\\citep{latalaoleszkiewicz2022}", text)
        text = text.replace(
            "``Taming under\nisoperimetry'' (2025) derives",
            r"\citep{tamingisoperimetry2025} derives"
        )
        text = text.replace(
            "Taming under isoperimetry\n    (2025)",
            r"\citep{tamingisoperimetry2025}"
        )
    return text


def convert_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    text, n_subs = convert_text(text)
    text = manual_fixups(text, path.name)
    path.write_text(text, encoding="utf-8")
    return n_subs


if __name__ == "__main__":
    chapters_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    total = 0
    for f in sorted(chapters_dir.glob("*.tex")):
        n = convert_file(f)
        if n:
            print(f"{f.name}: {n} citations converted")
        total += n
    print(f"\nTotal: {total} citations converted")
