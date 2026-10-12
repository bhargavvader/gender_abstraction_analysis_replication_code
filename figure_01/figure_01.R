# ==============================================================================
# figure_01.R  -- FIGURE 1 (fig1_text_data)
#   A  Human Judgments of Concreteness          (conc_brysbaert;       words + phrases)
#   B  AI Judgments of Concreteness             (conc_gpt_prob;        phrases only)
#   C  Human Judgments of Sensorimotor Strength (lanc_mean_perceptual; words only)
#
# MODEL SET: the 12-model registry shared with Figs. 2, 4c and 5 (registry.py), in all
#   three panels, scored in the rebuilt master (Wikipedia GloVe single-word scores
#   included: 31,609 of the 39,953 words are in GloVe's vocabulary).
# CONVENTIONS
#   * Panel C is the word-level run (12 lines; inset "N = 39,706 words"), matching the
#     main text; the averaged-phrase extension is in the SI (si_03_04). Set
#     panels$C$scope <- c("word", "phrase") for the pooled panel.
#   * Legend: 3 rows x 4 columns, 16-pt text; labels word2vec, GPT2, GPT4, Qwen3 (no
#     parentheticals) so Figs. 1 and 2 use the same names.
#   * Panels A and B share an identical y-axis range (both are 1-5 concreteness
#     ratings); panel C keeps its own range (Lancaster perceptual strength).
#
# Decile means and inset Ns are recomputed from the FULL master file (the one
# with the gender_* columns). The per-model correlations for the same registry
# are in fig1_per_model_correlations.csv (Python, same ntile deciles).
#
# PATHS: data/text/master/MASTER_all_models_gender_all_concreteness.csv under
# the repo root; results to results/figure_01/. Override with GC_REPO_DIR,
# GC_MASTER_FILE, GC_RESULTS_DIR. Run from anywhere with Rscript.
#
# Requires patchwork >= 1.2.0 (for axis_titles = "collect").
# ==============================================================================
rm(list = ls()); gc()
library(dplyr); library(tidyr); library(readr); library(tibble)
library(ggplot2); library(patchwork); library(grid)

# ============================== CONFIG ========================================
script_dir  <- tryCatch({ a <- commandArgs(FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)]); dirname(normalizePath(f)) },
                        error = function(e) getwd())
repo_dir    <- Sys.getenv("GC_REPO_DIR", dirname(script_dir))
master_file <- Sys.getenv("GC_MASTER_FILE", file.path(repo_dir, "data/text/master/MASTER_all_models_gender_all_concreteness.csv"))
results_dir <- Sys.getenv("GC_RESULTS_DIR", file.path(repo_dir, "results/figure_01/"))
out_stub    <- "fig1_text_data"       # -> fig1_text_data.png / .pdf (the names used in results.tex)

# display name -> gender column in the master (legend order = this order).
# Model set shared with Fig. 2 (fig2_by_domain). Keep the two registries in sync.
registry <- tibble::tribble(
  ~name,                   ~master_col,
  "FastText",              "gender_FastText",
  "word2vec",              "gender_word2vec_Original",   # Google News vectors
  "Wikipedia GloVe",       "gender_Wikipedia_GloVe",
  "ConceptNet",            "gender_ConceptNet",
  "BERT",                  "gender_BERT",
  "distilBERT",            "gender_distilBERT",
  "RoBERTa",               "gender_RoBERTa",
  "MPNet",                 "gender_MPNet",
  "GPT4",                  "gender_OpenAI_GPT4",         # text-embedding-3-large
  "Qwen3",                 "gender_Qwen3_Embedding",
  "Nomic v2 MoE",          "gender_Nomic_v2_MoE",
  "GPT2",                  "gender_GPT2_2026"
)

# panel definitions. `scope` = which master rows feed the panel.
panels <- list(
  A = list(title   = "Human Judgments\nof Concreteness",
           measure = "conc_brysbaert",
           scope   = c("word", "phrase"),
           ylab    = "Mean Concreteness Rating\n(1\u20135)"),
  B = list(title   = "AI Judgments\nof Concreteness",
           measure = "conc_gpt_prob",
           scope   = c("phrase"),
           ylab    = "Mean Concreteness Rating\n(GPT Norms, 1\u20135)"),
  C = list(title   = "Human Judgments of\nSensorimotor Strength",
           measure = "lanc_mean_perceptual",
           scope   = c("word"),                 # c("word", "phrase") for the pooled panel (SI)
           ylab    = "Mean Perceptual Strength\n(Lancaster Norms)")
)

MIN_TERMS  <- 100                      # skip a model in a panel if it scores fewer terms than this
X_LAB      <- "Gender Score Decile"    # direction is carried by the arrows under each panel
DIR_LEFT   <- "More Feminine"          # arrow labels under each panel (decile 1 side / decile 10 side)
DIR_RIGHT  <- "More Masculine"
DIR_DROP   <- 40                       # pt below the panel: centre line of the arrow row (labels sit one row lower)
DIR_SIZE   <- 16                       # pt, arrow-label font size
LEG_POINT  <- 6; LEG_LINE <- 2.4       # legend key glyph sizes (panel geoms are unchanged)
LEG_NROW   <- 3; LEG_TEXT <- 16        # 12 entries -> 3 x 4
FIG_W <- 16; FIG_H <- 9.2; DPI <- 400
# ============================ END CONFIG ======================================

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
model_levels <- registry$name
pal <- setNames(scales::hue_pal()(length(model_levels)), model_levels)  # one colour per model, all panels

# ---- data --------------------------------------------------------------------
master <- read_csv(master_file, show_col_types = FALSE, guess_max = 1e5)

one_panel <- function(tag) {
  p <- panels[[tag]]
  d <- master %>% filter(dataset %in% p$scope) %>%
    mutate(value = suppressWarnings(as.numeric(.data[[p$measure]]))) %>%
    filter(!is.na(value))
  dec <- lapply(seq_len(nrow(registry)), function(i) {
    col <- registry$master_col[i]
    if (!col %in% names(d)) stop("gender column '", col, "' not found in master file")
    x <- d %>% transmute(gender = suppressWarnings(as.numeric(.data[[col]])), value) %>%
      filter(!is.na(gender))
    if (nrow(x) < MIN_TERMS) { message("  panel ", tag, ": skipping ", registry$name[i],
                                       " (", nrow(x), " terms)"); return(NULL) }
    x %>% mutate(decile = ntile(gender, 10)) %>% group_by(decile) %>%
      summarise(value = mean(value), n_terms = n(), .groups = "drop") %>%
      mutate(panel = tag, model = registry$name[i])
  }) %>% bind_rows()
  # N = distinct terms with the measure AND a gender score from >= 1 plotted model
  used   <- registry$master_col[registry$name %in% unique(dec$model)]
  scored <- rowSums(!is.na(d[, used, drop = FALSE])) > 0
  n      <- table(factor(d$dataset[scored], levels = p$scope))
  list(dec = dec, n = setNames(as.integer(n), names(n)))
}
res     <- lapply(names(panels), one_panel); names(res) <- names(panels)
deciles <- bind_rows(lapply(res, `[[`, "dec"))
Ns      <- lapply(res, `[[`, "n")
write_csv(deciles, file.path(results_dir, paste0(out_stub, "_decile_means.csv")))
deciles$model <- factor(deciles$model, levels = model_levels)
# panels A and B are both concreteness ratings (1-5): give them an identical y-axis range
ab_ylim <- range(deciles$value[deciles$panel %in% c("A", "B")])
ab_ylim <- ab_ylim + c(-1, 1) * 0.03 * diff(ab_ylim)

# ---- plotting ----------------------------------------------------------------
n_label <- function(n) {
  unit <- c(word = "words", phrase = "phrases")
  paste(sprintf("N = %s %s", format(n, big.mark = ","), unit[names(n)]), collapse = "\n")
}

fig_theme <- theme_bw(base_size = 18) + theme(
  plot.title       = element_text(size = 21, hjust = 0.5, lineheight = 1.05),
  plot.tag         = element_text(size = 32, face = "bold"),
  axis.text        = element_text(size = 18, colour = "grey20"),
  axis.title.y     = element_text(size = 20, lineheight = 1.05),
  axis.title.x     = element_text(size = 22, margin = margin(t = 55)),   # leaves room for the direction key
  panel.grid.minor = element_blank(),
  axis.line        = element_line(colour = "black"),
  legend.position  = "bottom",
  legend.title     = element_blank(),
  legend.text      = element_text(size = LEG_TEXT, margin = margin(r = 12)),
  legend.key.width = unit(2.2, "lines"),
  legend.key.height = unit(1.4, "lines"),
  legend.margin    = margin(t = 2),
  plot.margin      = margin(6, 14, 4, 6))

# Direction key under each panel, between the tick labels and the x title:
# outward-pointing arrows on one row, their labels on the row beneath. Offsets are
# in absolute units (pt) so the key is identical in every panel.
direction_grob <- function() {
  y_arrow <- unit(0, "npc") - unit(DIR_DROP, "pt")
  y_text  <- y_arrow - unit(DIR_SIZE + 3, "pt")
  gp_a <- gpar(col = "grey20", fill = "grey20", lwd = 2.6)
  gp_t <- gpar(col = "grey20", fontsize = DIR_SIZE)
  ar   <- arrow(length = unit(9, "pt"), type = "closed", angle = 25)
  grobTree(
    segmentsGrob(unit(0.45, "npc"), y_arrow, unit(0.02, "npc"), y_arrow, arrow = ar, gp = gp_a),
    segmentsGrob(unit(0.55, "npc"), y_arrow, unit(0.98, "npc"), y_arrow, arrow = ar, gp = gp_a),
    textGrob(DIR_LEFT,  x = unit(0.235, "npc"), y = y_text, hjust = 0.5, gp = gp_t),
    textGrob(DIR_RIGHT, x = unit(0.765, "npc"), y = y_text, hjust = 0.5, gp = gp_t))
}

make_panel <- function(tag) {
  p <- panels[[tag]]
  ggplot(filter(deciles, panel == tag),
         aes(decile, value, colour = model, group = model)) +
    geom_line(linewidth = 1.05, show.legend = TRUE) +
    geom_point(size = 2.9, show.legend = TRUE) +
    annotate("label", x = Inf, y = Inf, label = n_label(Ns[[tag]]),
             hjust = 1.06, vjust = 1.2, size = 6.2, lineheight = 1.05,
             label.padding = unit(0.45, "lines"), label.r = unit(0.1, "lines"),
             label.size = 0.35, fill = "white", colour = "black") +
    annotation_custom(direction_grob()) +
    coord_cartesian(clip = "off", ylim = if (tag %in% c("A", "B")) ab_ylim else NULL) +
    scale_x_continuous(breaks = 1:10) +
    scale_colour_manual(values = pal, limits = model_levels, drop = FALSE,
                        guide = guide_legend(nrow = LEG_NROW, byrow = TRUE,
                                             override.aes = list(size = LEG_POINT, linewidth = LEG_LINE))) +
    labs(title = p$title, x = X_LAB, y = p$ylab, colour = NULL) +
    fig_theme
}

fig <- (make_panel("A") | make_panel("B") | make_panel("C")) +
  plot_layout(guides = "collect", axis_titles = "collect") +
  plot_annotation(tag_levels = "A") &
  theme(legend.position = "bottom")

ggsave(file.path(results_dir, paste0(out_stub, ".png")), fig, width = FIG_W, height = FIG_H, dpi = DPI)
ggsave(file.path(results_dir, paste0(out_stub, ".pdf")), fig, width = FIG_W, height = FIG_H, device = cairo_pdf)
cat("Saved to", results_dir, "\n"); print(Ns)
