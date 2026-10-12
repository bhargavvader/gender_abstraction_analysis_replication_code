"""figure_02_a_domains_flags_classes.py  (= fig2_by_domain/01_domains_flags_classes.py, 5 Oct 2026)

Fig. 2 pipeline, step 1 (same rules as the Sept 2026 scripts 01_build_domains.py, 03_exclusion_ladder_and_checks.py,
06_construal_drilldown.py, consolidated):
  * WordNet supersense (lexname) of each term's PRIMARY sense, gated on Brysbaert's dominant POS for words
    (ungated first-sense fallback when Brysbaert has no POS); phrases via their own WordNet entry, else head noun.
  * Exclusion flags: function/non-content words (Brysbaert Dom_Pos), lexically gendered terms (curated tokens and roots,
    -man/-woman/-ess/-ette morphology, first names in phrases, WordNet hypernym closure through male/female, sexed anatomy),
    and person-denoting nouns (noun.person).
  * Construal class (word form): words by Dom_Pos (Verb / Adverb / Adjective / Noun: event-state / Noun: entity);
    phrases by the POS of their WordNet entry, else NLTK tagger.
Writes (to results/figure_02/):
  vocab_L3_classified.csv      the Fig. 2 vocabulary with wn_lexname, cls and the gender_* columns
  all_terms_domains_flags.csv  domains + exclusion flags for EVERY term (used by si_05_lexical_covariates.py)

Inputs: the FULL master and the Brysbaert norms (for Dom_Pos). NLTK resources: wordnet, names, averaged_perceptron_tagger_eng.
"""
import os, re, sys, warnings, numpy as np, pandas as pd
from nltk.corpus import wordnet as wn, names
from nltk import pos_tag
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from paths import MASTER, BRYS, results
from registry import REGISTRY
OUT = results("figure_02")
GCOLS = [c for _, c in REGISTRY]
CONC = "conc_brysbaert"

df = pd.read_csv(MASTER, low_memory=False, keep_default_na=False, na_values=["", "NA", "NaN"])
for c in [CONC] + GCOLS: df[c] = pd.to_numeric(df[c], errors="coerce")
brys = pd.read_csv(BRYS)[["Word", "Dom_Pos"]]          # default NA handling: "#N/A" -> missing POS (ungated fallback), as in the Sept pipeline
brys["Word"] = brys["Word"].astype(str).str.strip().str.lower(); brys = brys.drop_duplicates("Word")
df["term_l"] = df["term"].astype(str).str.strip().str.lower()
df = df.merge(brys.rename(columns={"Word": "term_l"}), on="term_l", how="left")

# ------------------------------------------------------------------ domains ----------------------------------
POS_MAP = {"Noun": ["n"], "Verb": ["v"], "Adjective": ["a", "s"], "Adverb": ["r"]}
def lexname_for(lemma, pos_list=None):
    syns = wn.synsets(lemma.replace(" ", "_"))
    if not syns: return np.nan, "none"
    if pos_list:
        sp = [s for s in syns if s.pos() in pos_list]
        return (sp[0].lexname(), "direct_pos") if sp else (np.nan, "pos_mismatch")
    return syns[0].lexname(), "direct"
def assign_word(row):
    dp = row["Dom_Pos"]
    if pd.isna(dp):
        lex, how = lexname_for(row["term_l"]); return lex, ("ungated_" + how if isinstance(lex, str) else "none")
    pos_list = POS_MAP.get(dp)
    if pos_list is None: return np.nan, "no_content_pos"
    return lexname_for(row["term_l"], pos_list)
def assign_phrase(term):
    lex, how = lexname_for(term)
    if isinstance(lex, str): return lex, "mwe_" + how
    toks = term.replace("-", " ").split()
    if len(toks) >= 2:
        syns = [s for s in wn.synsets(toks[-1]) if s.pos() == "n"]
        if syns: return syns[0].lexname(), "head_noun"
    return np.nan, "none"
print("Assigning WordNet domains..."); out = []
for _, r in df.iterrows():
    out.append(assign_word(r) if r["dataset"] == "word" else assign_phrase(r["term_l"]))
df["wn_lexname"], df["wn_lex_method"] = zip(*out)

# ------------------------------------------------------------------ flags ------------------------------------
GENDER_TOKENS = set("""man woman men women male female males females boy boys girl girls he she him her his hers
himself herself guy guys gal gals dude dudes bloke chap fellow lad lads lass lasses mr mrs ms miss mister missus
sir madam maam gentleman gentlemen lady ladies king kings queen queens prince princess lord lords lordship father
fathers mother mothers dad dads daddy mom moms mommy mum mummy papa mama husband husbands wife wives son sons daughter
daughters brother brothers sister sisters uncle uncles aunt aunts auntie nephew niece grandfather grandmother grandpa
grandma granny grandson granddaughter bride brides groom bridegroom widow widower widows boyfriend girlfriend mistress
master masters monk monks nun nuns priest priestess pope bishop rabbi imam god gods goddess goddesses lord jesus christ
mohammed muhammad allah buddha messiah devil satan hero heroes heroine heroines chick chicks babe babes hen hens cow cows
mare mares sow sows ewe ewes doe vixen bitch bull bulls stallion stallions rooster cock cocks ram rams stag stags buck
bucks gelding drake gander tomcat tenor baritone soprano alto contralto macho manly manliness womanly womanhood manhood
mankind womankind feminine femininity masculine masculinity effeminate virile virility pregnant pregnancy maternal
paternal maternity paternity motherly fatherly motherhood fatherhood ladylike girlish boyish girly matron matriarch
patriarch patriarchy matriarchy feminism feminist feminists sexism sexist misogyny misogynist misandry lesbian lesbians
gay gays homosexual transgender transsexual bisexual eunuch harem concubine courtesan geisha hooker prostitute whore
slut spinster bachelor bachelors batchelor bachelorette dame damsel maiden maid maids wench belle diva emperor empress
duke duchess count countess baron baroness earl marquis marquise czar tsar tsarina nanny nannies au-pair midwife
midwives nursemaid housewife housewives househusband stewardess actress actresses waitress waitresses hostess
seamstress governess sorceress enchantress temptress witch witches wizard warlock bro sis mama papa boyhood girlhood
manpower men's women's boy's girl's ladies' gents""".split())
PRONOUN_LIKE = {"he'd", "he's", "he'll", "she'd", "she's", "she'll", "him.", "her."}
BODY_KW = ("breast", "boob", "bosom", "vagina", "vulva", "uterus", "uterine", "womb", "ovar", "cervix", "cervical",
           "clitor", "labia", "penis", "penile", "testic", "testes", "scrot", "prostat", "menstru", "menopaus", "semen",
           "sperm", "erection", "erectile", "lactat", "breastfeed", "nipple", "beard", "mustache", "moustache",
           "goatee", "hymen", "phallus", "phallic", "bra ", " bra", "brassiere", "ovulat", "gestat", "childbirth", "midwif")
MAN_EXCEPT = {"human", "humans", "roman", "romans", "german", "germans", "ottoman", "shaman", "shamans", "talisman",
              "abdomen", "specimen", "regimen", "amen", "omen", "semen", "lumen", "acumen", "cayman", "brahman",
              "ramen", "yemen", "carmen", "hymen", "bitumen", "cyclamen", "albumen", "stamen", "tegmen", "women"}
ESS_EXCEPT = {"process", "dress", "mess", "less", "bless", "chess", "press", "stress", "guess", "access", "excess",
              "recess", "success", "abscess", "confess", "assess", "possess", "address", "progress", "regress",
              "congress", "express", "impress", "witness", "business", "duress", "largess", "noblesse", "cress",
              "tress", "egress", "ingress", "digress", "compress", "depress", "repress", "suppress", "obsess",
              "finesse", "chess", "prowess", "harness", "fortress", "buttress", "mattress", "cutlass", "cypress",
              "bourgeois"}
MALE_NAMES = {n.lower() for n in names.words("male.txt")}; FEMALE_NAMES = {n.lower() for n in names.words("female.txt")}
NAME_SET = (MALE_NAMES | FEMALE_NAMES) - {"will", "may", "june", "rose", "grace", "hope", "faith", "joy", "bill",
              "mark", "art", "pat", "sue", "guy", "jack", "jimmy", "chase", "rich", "rob", "don", "van", "ray", "jay",
              "gay", "chip", "duke", "earl", "baron", "wade", "cliff", "clay", "dick", "frank", "gene", "grant",
              "hazel", "holly", "iris", "ivy", "jade", "jasmine", "lily", "olive", "pearl", "ruby", "violet",
              "amber", "april", "autumn", "dawn", "eve", "summer", "crystal", "daisy", "flora", "heather",
              "misty", "penny", "robin", "sandy", "star", "brandy", "sherry", "candy", "ginger", "honey",
              "kitty", "candy", "sunny", "wendy", "molly", "polly", "dolly", "cherry", "lynn", "lee", "kim",
              "kay", "dee", "bea", "di", "jo", "al", "ed", "ty", "cy", "sol", "hal", "lou", "les", "ira", "bob"}
NAME_SET = {n for n in NAME_SET if len(n) >= 4 and not wn.synsets(n)} | {
    "jack", "joe", "john", "jim", "jimmy", "mike", "michael", "bob", "tom", "dick", "harry", "george", "peter",
    "paul", "adam", "eve", "mary", "jane", "jill", "jenny", "sally", "alice", "anna", "betty", "jones", "robinson",
    "fixit", "mohammed", "moses", "abraham", "noah", "david", "goliath", "solomon", "samson", "delilah", "judas",
    "lazarus", "cain", "abel", "jesus", "christ", "peter", "simon", "thomas", "james", "andrew", "sam", "fred",
    "murphy", "riley", "hobson", "morton", "parkinson", "sod", "hooray", "harry", "jerry", "larry", "moe", "curly"}
FEM_SYN = {"female.n.01", "female.n.02", "woman.n.01", "girl.n.01", "female_child.n.01", "mother.n.01"}
MASC_SYN = {"male.n.01", "male.n.02", "man.n.01", "boy.n.01", "male_child.n.01", "father.n.01"}
GENDER_ROOTS = ("father","mother","brother","sister","husband","wife","woman","women","girl","boy","lady","ladies",
                "lord","queen","prince","macho","manly","manli","womanly","sportsman","gentleman","bastard","virgin",
                "maiden","bride","groom","widow","masculin","feminin","matern","patern","matriarch","patriarch",
                "feminis","misogyn","sexis","lesbian","uncle","aunt","nephew","niece","daughter","mankind","manhood",
                "womanhood","boyish","girlish","effeminate","virile","hero","heroine","goddess","godlike","godly",
                "mister","madam","mistress","dame","damsel","wench","spinster","bachelor","eunuch","harem","actress",
                "waitress","hostess","stewardess","seamstress","governess","princess","duchess","countess","baroness",
                "empress","matron","nanny","nurse","midwif","housewi","househusband","pregnan","chick","babe")
ROOT_EXCEPT = ("boycott","boysenberry","malady","smother","chicken","chickpea","heroic","heroism","heron","dame ",
               "nurser","nursing home","nursery","nursemaid","nurse shark")
def first_noun_closure(lemma):
    syns = wn.synsets(lemma.replace(" ", "_"), pos="n")
    if not syns: return set()
    s = syns[0]; return {h.name() for h in s.closure(lambda x: x.hypernyms())} | {s.name()}
def tokens(term):
    t = str(term).lower().replace("\u2019", "'"); t = re.sub(r"'s\b", "", t)
    return [x for x in re.split(r"[\s_\-/,.;:!?()\"]+", t) if x]
def lexical_gender(row):
    term = str(row["term"]).lower(); toks = tokens(term)
    if any(r in term for r in GENDER_ROOTS) and not any(e in term for e in ROOT_EXCEPT): return True
    if set(toks) & GENDER_TOKENS or set(toks) & PRONOUN_LIKE: return True
    if row["dataset"] == "phrase" and set(toks) & NAME_SET: return True
    for tok in toks:
        if (tok.endswith("man") or tok.endswith("men")) and tok not in MAN_EXCEPT and len(tok) > 4: return True
        if tok.endswith("woman") or tok.endswith("women"): return True
    if row["wn_lexname"] == "noun.person":
        for tok in toks:
            if (tok.endswith("ess") and tok not in ESS_EXCEPT and len(tok) > 5) or tok.endswith("ette") or tok.endswith("trix"):
                return True
    if any(k in (" " + term + " ") for k in BODY_KW): return True
    cl = first_noun_closure(term) if row["dataset"] == "word" or str(row["wn_lex_method"]).startswith("mwe") \
        else first_noun_closure(toks[-1]) if toks else set()
    return bool(cl & (FEM_SYN | MASC_SYN))
NONCONTENT = {"Name", "Preposition", "Number", "Pronoun", "Interjection", "Conjunction", "Determiner",
              "Article", "Letter", "Ex", "To", "Not", "Unclassified", "#N/A"}
print("Computing exclusion flags...")
df["flag_lexical_gender"] = df.apply(lexical_gender, axis=1)
df["flag_person"] = df["wn_lexname"].eq("noun.person")
df["flag_noncontent"] = np.where(df.dataset == "word",
                                 df["Dom_Pos"].isin(NONCONTENT) | (df["Dom_Pos"].isna() & df["wn_lexname"].isna()), False)

# ------------------------------------------------------------------ L3 vocabulary + construal classes ----------
scored_any = df[GCOLS].notna().any(axis=1)
L3 = df[CONC].notna() & scored_any & ~df.flag_noncontent & ~df.flag_lexical_gender & ~df.flag_person
EVENTIVE = {"noun.act", "noun.event", "noun.process", "noun.cognition", "noun.communication", "noun.motive",
            "noun.state", "noun.feeling", "noun.attribute", "noun.relation", "noun.phenomenon", "noun.time",
            "noun.possession", "noun.quantity"}
ENTITY = {"noun.artifact", "noun.object", "noun.substance", "noun.food", "noun.body", "noun.plant", "noun.animal",
          "noun.location", "noun.group", "noun.shape", "noun.Tops"}
def word_class(r):
    pos = r["Dom_Pos"]; lex = r["wn_lexname"] if isinstance(r["wn_lexname"], str) else ""
    if pos == "Verb" or (pd.isna(pos) and lex.startswith("verb.")): return "Verb"
    if pos == "Adverb" or (pd.isna(pos) and lex.startswith("adv.")): return "Adverb"
    if pos == "Adjective" or (pd.isna(pos) and lex.startswith("adj.")): return "Adjective"
    if pos == "Noun" or (pd.isna(pos) and lex.startswith("noun.")):
        if lex in EVENTIVE: return "Noun: event/state"
        if lex in ENTITY: return "Noun: entity"
        return "Noun: other"
    return np.nan
def phrase_class(r):
    term = str(r["term"]).lower(); lex = r["wn_lexname"] if isinstance(r["wn_lexname"], str) else ""
    if str(r["wn_lex_method"]).startswith("mwe"):
        s = wn.synsets(term.replace(" ", "_")); p = s[0].pos() if s else None
        if p == "v": return "Verb phrase / clause"
        if p in ("a", "s", "r"): return "Modifier phrase"
        if p == "n": return "Noun phrase: event/state" if lex in EVENTIVE else "Noun phrase: entity" if lex in ENTITY else "Noun phrase: other"
    toks = term.replace("\u2019", "'").split(); tags = [t for _, t in pos_tag(toks)]
    if any(t.startswith("VB") for t in tags) and not tags[-1].startswith("NN") or (tags and tags[0].startswith("VB")):
        return "Verb phrase / clause"
    if any(t.startswith("VB") for t in tags): return "Verb phrase / clause"
    if tags and tags[-1].startswith("NN"):
        return "Noun phrase: event/state" if lex in EVENTIVE else "Noun phrase: entity" if lex in ENTITY else "Noun phrase: other"
    if tags and (tags[-1].startswith("JJ") or tags[-1].startswith("RB") or tags[0].startswith("IN")): return "Modifier phrase"
    return "Noun phrase: other"
# full table (all terms) for the SI robustness analyses: domains + exclusion flags
df[["term", "term_l", "dataset", "Dom_Pos", "wn_lexname", "wn_lex_method", "flag_noncontent", "flag_lexical_gender", "flag_person"]].to_csv(
    os.path.join(OUT, "all_terms_domains_flags.csv"), index=False)
print("Classifying construal...")
v = df[L3].copy()
v["cls"] = np.where(v.dataset == "word", v.apply(word_class, axis=1), None)
v.loc[v.dataset == "phrase", "cls"] = v[v.dataset == "phrase"].apply(phrase_class, axis=1)
keep = ["term", "term_l", "dataset", CONC, "Dom_Pos", "wn_lexname", "wn_lex_method", "cls"] + GCOLS
v[keep].to_csv(os.path.join(OUT, "vocab_L3_classified.csv"), index=False)
for ds in ["word", "phrase"]:
    s = v[v.dataset == ds]
    print(f"{ds}: L3 vocabulary N={len(s):,}; domain assigned {s.wn_lexname.notna().mean():.1%}; classes: {s.cls.value_counts().to_dict()}")
for f in ["flag_noncontent", "flag_lexical_gender", "flag_person"]:
    print(f"  {f}: words={df[(df.dataset=='word')][f].sum():,}  phrases={df[(df.dataset=='phrase')][f].sum():,}")
