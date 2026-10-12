# ==============================================================================
# si_02_image_face_count_robustness.R  (= 04_image_robustness_controls.R, 21 Sept 2026)
#
# SUPPLEMENTARY SECTION 2 / SUPPLEMENTARY TABLE 2
# IMAGE DATA -- ROBUSTNESS OF THE ABSTRACT -> MALE ASSOCIATION TO FACE-COUNT CONTROLS
#
# Outcome : frac_male (fraction of detected faces classified male) per search term.
#           The image file codes frac_male = 0 for terms with NO detected faces;
#           the main figure averages over all terms. The specifications below test
#           whether the negative association survives when that coding, and
#           variation in the number of faces, are handled in different ways.
# Predictors (run separately): conc (Brysbaert concreteness) and perc (Lancaster
#           perceptual strength, mean of six modalities).
# All continuous variables are z-scored within each estimation sample, so OLS
# coefficients are standardized betas; SEs are heteroskedasticity-robust (HC1).
#
# Writes (results/si_02/): image_robustness_results.csv, image_robustness_table_rows.tex,
#         image_noface_by_bin.csv
# Requires: dplyr, readr, tidyr, sandwich, lmtest
# PATHS: data/images/image_class_w_conc_PROCESSED.csv, data/norms/lancaster_sensorimotor_norms_2019.csv
#        under the repo root; override with GC_REPO_DIR, GC_IMAGE_FILE, GC_LANC_FILE, GC_RESULTS_DIR.
# ==============================================================================
rm(list = ls()); gc()
suppressMessages({library(dplyr); library(readr); library(sandwich); library(lmtest)})

# ============================== CONFIG ========================================
script_dir  <- tryCatch({ a <- commandArgs(FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)]); dirname(normalizePath(f)) },
                        error = function(e) getwd())
repo_dir    <- Sys.getenv("GC_REPO_DIR", dirname(script_dir))
image_file  <- Sys.getenv("GC_IMAGE_FILE", file.path(repo_dir, "data/images/image_class_w_conc_PROCESSED.csv"))
lanc_file   <- Sys.getenv("GC_LANC_FILE",  file.path(repo_dir, "data/norms/lancaster_sensorimotor_norms_2019.csv"))
results_dir <- Sys.getenv("GC_RESULTS_DIR", file.path(repo_dir, "results/si_02/"))
FACE_THRESHOLDS <- c(5, 10, 20)
N_BINS <- 20
# ============================ END CONFIG ======================================
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

perc_cols <- paste0(c("Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual"), ".mean")
lanc <- read_csv(lanc_file, show_col_types = FALSE) %>%
  mutate(key = tolower(trimws(Word)), perc = rowMeans(across(all_of(perc_cols)))) %>%
  distinct(key, .keep_all = TRUE) %>% select(key, perc)
img <- read_csv(image_file, show_col_types = FALSE, guess_max = 1e5) %>%
  mutate(key = tolower(trimws(query)), has_face = as.integer(num_faces > 0),
         log_faces = log1p(num_faces)) %>%
  left_join(lanc, by = "key")
stopifnot(all(img$num_males + img$num_females == img$num_faces))

z <- function(v) as.numeric(scale(v))
rows <- list()
add <- function(pred, id, spec, stat, est, lo, hi, p, n) {
  rows[[length(rows) + 1]] <<- tibble(predictor = pred, id = id, specification = spec, statistic = stat,
                                     estimate = est, ci_low = lo, ci_high = hi, p = p, n = n)
}
ols <- function(pred, id, spec, d, rhs = "", weights = NULL, y = "frac_male") {
  d <- d[!is.na(d[[pred]]), ]
  d$y <- z(d[[y]]); d$x <- z(d[[pred]])
  d$w_raw <- if (is.null(weights)) 1 else d[[weights]]          # keep raw weights before z-scoring
  for (v in c("num_faces", "log_faces", "freq")) d[[v]] <- z(d[[v]])
  m  <- lm(as.formula(paste("y ~ x", rhs)), d, weights = w_raw)
  ct <- coeftest(m, vcov. = vcovHC(m, type = "HC1"))
  ci <- coefci(m, vcov. = vcovHC(m, type = "HC1"))["x", ]
  add(pred, id, spec, "beta", ct["x", 1], ci[1], ci[2], ct["x", 4], nobs(m))
}

for (pred in c("conc", "perc")) {
  all_t  <- img
  face_t <- filter(img, num_faces >= 1)
  # --- A. all search terms (zero-face terms coded 0, as in the main figure)
  ols(pred, "A1", "All terms; no controls (main figure)", all_t)
  ols(pred, "A2", "All terms; + number of faces", all_t, "+ num_faces")
  ols(pred, "A3", "All terms; + log(1 + faces)", all_t, "+ log_faces")
  ols(pred, "A4", "All terms; + any-face indicator", all_t, "+ has_face")
  ols(pred, "A5", "All terms; + any-face indicator + log(1 + faces)", all_t, "+ has_face + log_faces")
  ols(pred, "A6", "All terms; + any-face indicator + log(1 + faces) + word frequency", all_t, "+ has_face + log_faces + freq")
  # --- B. restricted to terms with detected faces
  ols(pred, "B1", "Terms with $\\geq 1$ face; no controls", face_t)
  ols(pred, "B2", "Terms with $\\geq 1$ face; + log(1 + faces)", face_t, "+ log_faces")
  ols(pred, "B3", "Terms with $\\geq 1$ face; + log(1 + faces) + word frequency", face_t, "+ log_faces + freq")
  for (k in FACE_THRESHOLDS)
    ols(pred, paste0("B", 3 + match(k, FACE_THRESHOLDS)),
        sprintf("Terms with $\\geq %d$ faces; + log(1 + faces) + word frequency", k),
        filter(img, num_faces >= k), "+ log_faces + freq")
  # --- C. alternative estimators
  ols(pred, "C1", "Terms with $\\geq 1$ face; weighted by number of faces", face_t, weights = "num_faces")
  d  <- face_t[!is.na(face_t[[pred]]), ]; d$x <- z(d[[pred]])
  qb <- glm(cbind(num_males, num_females) ~ x, family = quasibinomial, data = d)
  s  <- summary(qb)$coefficients["x", ]
  add(pred, "C2", "Face-level quasi-binomial logit (odds ratio per s.d.)", "OR",
      exp(s[1]), exp(s[1] - 1.96 * s[2]), exp(s[1] + 1.96 * s[2]), s[4], nrow(d))
  sp <- suppressWarnings(cor.test(d[[pred]], d$frac_male, method = "spearman"))
  rho <- unname(sp$estimate); se <- 1 / sqrt(nrow(d) - 3)            # Fisher-z interval
  add(pred, "C3", "Terms with $\\geq 1$ face; Spearman rank correlation", "rho",
      rho, tanh(atanh(rho) - 1.96 * se), tanh(atanh(rho) + 1.96 * se), sp$p.value, nrow(d))
  # --- D. outcome for which zero-face terms are neutral rather than 0% male
  ols(pred, "D1", "All terms; outcome = male minus female fraction", all_t, y = "gender_parity")
  # --- E. the no-face rate itself
  d  <- all_t[!is.na(all_t[[pred]]), ]; d$x <- z(d[[pred]]); d$no_face <- 1 - d$has_face
  lg <- glm(no_face ~ x, family = binomial, data = d); s <- summary(lg)$coefficients["x", ]
  add(pred, "E1", "All terms; outcome = no face detected (logit, odds ratio per s.d.)", "OR",
      exp(s[1]), exp(s[1] - 1.96 * s[2]), exp(s[1] + 1.96 * s[2]), s[4], nrow(d))
}

res <- bind_rows(rows)
write_csv(res, file.path(results_dir, "image_robustness_results.csv"))

# descriptive: share of terms with no detected face, by bin of each predictor
nf <- bind_rows(lapply(c("conc", "perc"), function(pred)
  img %>% filter(!is.na(.data[[pred]])) %>% mutate(bin = ntile(.data[[pred]], N_BINS)) %>%
    group_by(bin) %>% summarise(mean_x = mean(.data[[pred]]), share_no_face = mean(1 - has_face),
                                mean_frac_male_all = mean(frac_male),
                                mean_frac_male_faces = mean(frac_male[num_faces > 0]), .groups = "drop") %>%
    mutate(predictor = pred, .before = 1)))
write_csv(nf, file.path(results_dir, "image_noface_by_bin.csv"))

# LaTeX rows (one table, the two predictors side by side)
fmt_p <- function(p) vapply(p, function(v) {
  if (v < 1e-300) return("$<10^{-300}$")
  if (v >= 0.001) return(sprintf("%.3f", v))
  e <- floor(log10(v)); m <- v / 10^e
  if (round(m, 1) >= 10) { m <- m / 10; e <- e + 1 }
  sprintf("$%.1f \\times 10^{%d}$", m, as.integer(e))
}, character(1))
fmt_e <- function(e, l, h) sprintf("$%s$ [$%s$, $%s$]", formatC(e, format = "f", digits = 3, flag = " "),
                                   formatC(l, format = "f", digits = 3), formatC(h, format = "f", digits = 3))
w <- res %>% mutate(cell = fmt_e(estimate, ci_low, ci_high), pf = fmt_p(p), nf = format(n, big.mark = ",")) %>%
  select(predictor, id, specification, cell, pf, nf) %>%
  tidyr::pivot_wider(names_from = predictor, values_from = c(cell, pf, nf))
lines <- sprintf("%s & %s & %s & %s & %s & %s & %s & %s \\\\", w$id, w$specification,
                 w$cell_conc, w$pf_conc, w$nf_conc, w$cell_perc, w$pf_perc, w$nf_perc)
writeLines(lines, file.path(results_dir, "image_robustness_table_rows.tex"))

print(as.data.frame(res %>% mutate(across(c(estimate, ci_low, ci_high), ~ round(.x, 4)))), row.names = FALSE)
cat("\nterms:", nrow(img), "| no face:", sum(img$has_face == 0), "| faces:", sum(img$num_faces),
    "| Lancaster-matched terms:", sum(!is.na(img$perc)), "\n")
