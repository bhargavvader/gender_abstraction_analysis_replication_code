# ==============================================================================
# figure_03.R  (= 03_fig_two_panel_image_pmale.R, final version of 21 Sept 2026)
#
# TWO-PANEL IMAGE FIGURE (fig3_image_results): P(Male Face in Google Image) by
#   A  human judgments of concreteness (Brysbaert Conc.M, `conc`)
#   B  Lancaster perceptual strength (mean of the six perceptual modalities)
#
# Each panel: search terms split into N_BINS equal-count bins on the x measure
# (ntile), one point per bin (mean x, mean frac_male), loess through the bin
# means with its 95% band -- the same recipe as the original single panels.
# Final conventions: each panel keeps its own x-range (MATCH_X = FALSE), the
# y-range is shared across panels (MATCH_Y = TRUE), both x subtitles read
# "(Human Judgments)", inset "N ≈ 12 million images" over the search-term count.
#
# IMPORTANT (WORD_SET): image_class_w_conc_PROCESSED.csv codes frac_male = 0 for
# search terms with NO detected faces (4,043 of 39,956). The main-text figure
# averages over ALL terms (WORD_SET = "all"), so those zeros pull the curve down
# wherever faceless terms are common (highly concrete terms). WORD_SET = "faces"
# restricts to terms with >= MIN_FACES detected faces, i.e. P(male | a face was
# detected); the SI (si_02 / si_08) shows the direction holds under that coding.
#
# PATHS: data/images/image_class_w_conc_PROCESSED.csv and
# data/norms/lancaster_sensorimotor_norms_2019.csv under the repo root; results
# to results/figure_03/. Override with GC_REPO_DIR, GC_IMAGE_FILE, GC_LANC_FILE,
# GC_RESULTS_DIR, GC_WORD_SET.
# ==============================================================================
rm(list = ls()); gc()
library(dplyr); library(readr); library(ggplot2); library(patchwork)

# ============================== CONFIG ========================================
script_dir  <- tryCatch({ a <- commandArgs(FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)]); dirname(normalizePath(f)) },
                        error = function(e) getwd())
repo_dir    <- Sys.getenv("GC_REPO_DIR", dirname(script_dir))
image_file  <- Sys.getenv("GC_IMAGE_FILE", file.path(repo_dir, "data/images/image_class_w_conc_PROCESSED.csv"))
lanc_file   <- Sys.getenv("GC_LANC_FILE",  file.path(repo_dir, "data/norms/lancaster_sensorimotor_norms_2019.csv"))
results_dir <- Sys.getenv("GC_RESULTS_DIR", file.path(repo_dir, "results/figure_03/"))

WORD_SET    <- Sys.getenv("GC_WORD_SET", "all")   # "all" = every search term (main figure) | "faces" = terms with >= MIN_FACES faces
MIN_FACES   <- 1
N_BINS      <- 20

MATCH_X     <- FALSE           # same x-axis range in both panels
X_LIMS      <- c(0.5, 5)       # used when MATCH_X (Brysbaert is 1-5, Lancaster 0-5)
MATCH_Y     <- TRUE            # same y-axis range in both panels
PARITY_LINE <- (WORD_SET == "all")   # horizontal line at 0.5 (off the scale in the "faces" version)

# Total images examined (faces or not). NOT derivable from the project files
# (they hold face counts only), so it is entered by hand; NA drops the line.
N_IMAGES_LABEL <- c(A = "N \u2248 12 million images", B = "N \u2248 12 million images")

Y_LAB  <- "P(Male Face in Google Image)"
X_LABS <- c(A = "Average Concreteness\n(Human Judgments)",
            B = "Average Perceptual Strength\n(Human Judgments)")
out_stub <- paste0("FIG_two_panel_image_pmale_", ifelse(WORD_SET == "all", "all-terms", "terms-with-faces"))
FIG_W <- 14; FIG_H <- 7.6; DPI <- 400
# ============================ END CONFIG ======================================

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# ---- data --------------------------------------------------------------------
img <- read_csv(image_file, show_col_types = FALSE, guess_max = 1e5) %>%
  mutate(key = tolower(trimws(query)))
perc_cols <- paste0(c("Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual"), ".mean")
lanc <- read_csv(lanc_file, show_col_types = FALSE) %>%
  mutate(key = tolower(trimws(Word)), perc = rowMeans(across(all_of(perc_cols)))) %>%
  distinct(key, .keep_all = TRUE) %>% select(key, perc)

if (WORD_SET == "faces") img <- filter(img, num_faces >= MIN_FACES)

dat <- list(A = img %>% transmute(key, x = conc, frac_male, num_faces) %>% filter(!is.na(x)),
            B = img %>% inner_join(lanc, by = "key") %>% transmute(key, x = perc, frac_male, num_faces) %>% filter(!is.na(x)))

binned <- lapply(dat, function(d) d %>% mutate(bin = ntile(x, N_BINS)) %>% group_by(bin) %>%
                   summarise(x = mean(x), p_male = mean(frac_male), n_terms = n(), .groups = "drop"))

for (t in names(dat)) {
  ct <- cor.test(dat[[t]]$x, dat[[t]]$frac_male)
  cat(sprintf("Panel %s [%s]: %s terms, %s faces; term-level r = %.3f [%.3f, %.3f]; bin means %.3f -> %.3f\n",
              t, WORD_SET, format(nrow(dat[[t]]), big.mark = ","), format(sum(dat[[t]]$num_faces), big.mark = ","),
              ct$estimate, ct$conf.int[1], ct$conf.int[2], binned[[t]]$p_male[1], binned[[t]]$p_male[N_BINS]))
}
write_csv(bind_rows(binned, .id = "panel"), file.path(results_dir, paste0(out_stub, "_bin_means.csv")))

# ---- plotting ----------------------------------------------------------------
n_label <- function(t) {
  lines <- c(if (!is.na(N_IMAGES_LABEL[[t]])) N_IMAGES_LABEL[[t]],
             sprintf("N = %s search terms", format(nrow(dat[[t]]), big.mark = ",")))
  paste(lines, collapse = "\n")
}

fig_theme <- theme_bw(base_size = 18) + theme(
  plot.tag         = element_text(size = 32, face = "bold"),
  axis.text        = element_text(size = 20, colour = "grey20"),
  axis.title.x     = element_text(size = 22, lineheight = 1.05, margin = margin(t = 10)),
  axis.title.y     = element_text(size = 22, margin = margin(r = 8)),
  panel.grid       = element_blank(),
  axis.line        = element_line(colour = "black"),
  plot.margin      = margin(6, 16, 6, 6))

y_all <- range(unlist(lapply(binned, `[[`, "p_male")), if (PARITY_LINE) 0.5)

# y tick spacing that always lands on two-decimal values
y_step <- function(t) {
  r <- if (MATCH_Y) diff(y_all) else diff(range(binned[[t]]$p_male, if (PARITY_LINE) 0.5))
  if (r > 0.15) 0.05 else if (r > 0.07) 0.02 else 0.01
}

make_panel <- function(t) {
  ggplot(binned[[t]], aes(x, p_male)) +
    { if (PARITY_LINE) geom_hline(yintercept = 0.5, linewidth = 0.6) } +
    geom_smooth(method = "loess", formula = y ~ x, se = TRUE, linewidth = 1.1,
                colour = "#3366FF", fill = "grey60", alpha = 0.4) +
    geom_point(size = 5.5) +
    annotate("label", x = Inf, y = Inf, label = n_label(t), hjust = 1.05, vjust = 1.2,
             size = 6.2, lineheight = 1.05, label.padding = unit(0.45, "lines"),
             label.r = unit(0.1, "lines"), label.size = 0.35, fill = "white") +
    { if (MATCH_X) scale_x_continuous(breaks = 1:5) } +
    coord_cartesian(xlim = if (MATCH_X) X_LIMS else NULL,
                    ylim = if (MATCH_Y) y_all + c(-0.08, 0.14) * diff(y_all) else NULL) +
    { if (PARITY_LINE && !MATCH_Y) expand_limits(y = 0.5) } +
    scale_y_continuous(breaks = scales::breaks_width(y_step(t)),
                       labels = function(v) sprintf("%.2f", v)) +
    labs(x = X_LABS[[t]], y = Y_LAB) +
    fig_theme
}

fig <- (make_panel("A") | make_panel("B")) + plot_annotation(tag_levels = "A")

ggsave(file.path(results_dir, paste0(out_stub, ".png")), fig, width = FIG_W, height = FIG_H, dpi = DPI)
ggsave(file.path(results_dir, paste0(out_stub, ".pdf")), fig, width = FIG_W, height = FIG_H, device = cairo_pdf)
cat("Saved", out_stub, "to", results_dir, "\n")
# The manuscript file fig3_image_results.{png,pdf} is the WORD_SET = "all" output of this script.
