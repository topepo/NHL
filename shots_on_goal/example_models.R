library(tidymodels)
library(doMC)
library(embed)

# ------------------------------------------------------------------------------

registerDoMC(cores = 20)
tidymodels_prefer()
theme_set(theme_bw())

# ------------------------------------------------------------------------------

load("shots_on_goal.RData")

# ------------------------------------------------------------------------------

set.seed(1)
val_split <- validation_split(shots_on_goal)
ct <- control_grid(parallel_over = "everything", save_pred = TRUE)

# ------------------------------------------------------------------------------

ranger_recipe <-
  recipe(formula = on_goal ~ ., data = shots_on_goal) %>%
  step_lencode_mixed(shooter, outcome = vars(on_goal))

ranger_spec <-
  rand_forest(mtry = tune(),
              min_n = tune(),
              trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

ranger_workflow <-
  workflow() %>%
  add_recipe(ranger_recipe) %>%
  add_model(ranger_spec)

set.seed(498)
ranger_tune <-
  tune_grid(
    ranger_workflow,
    resamples = val_split,
    grid = 10,
    control = ct
  )

# ------------------------------------------------------------------------------

set.seed(498)
rpart_tune <-
  decision_tree(cost_complexity = tune(), min_n = tune()) %>%
  set_mode("classification") %>%
  tune_grid(ranger_recipe,
            resamples = val_split,
            grid = 20,
            control = ct)

# ------------------------------------------------------------------------------


glmnet_recipe <-
  recipe(formula = on_goal ~ ., data = shots_on_goal) %>%
  step_lencode_mixed(shooter, outcome = vars(on_goal)) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors(), -all_nominal())

glmnet_spec <-
  logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

glmnet_workflow <-
  workflow() %>%
  add_recipe(glmnet_recipe) %>%
  add_model(glmnet_spec)

glmnet_grid <-
  tidyr::crossing(
    penalty = 10 ^ seq(-6, -1, length.out = 20),
    mixture = c(0.05, 0.2, 0.4, 0.6, 0.8, 1)
  )

glmnet_tune <-
  tune_grid(
    glmnet_workflow,
    resamples = val_split,
    grid = glmnet_grid,
    control = ct
  )

# ------------------------------------------------------------------------------

show_best(ranger_tune)
show_best(rpart_tune)
show_best(glmnet_tune)

# ------------------------------------------------------------------------------

# https://thewincolumn.ca/2021/01/15/r-tutorial-creating-an-nhl-rink-using-the-tidyverse/
devtools::source_url("https://raw.githubusercontent.com/mrbilltran/the-win-column/master/nhl_rink_plot.R")

NHL_red <- NHL_blue <- NHL_light_blue <- rgb(0, 0, 0, .2)

# ------------------------------------------------------------------------------

rpart_pred <-
  rpart_tune %>%
  collect_predictions(parameters = select_best(rpart_tune, metric = "rocc_auc")) %>%
  mutate(
    prediction = ifelse(.pred_class == on_goal, "correct", "error"),
    prediction = factor(prediction, levels = c("error", "correct"))
    ) %>%
  inner_join(shots_on_goal %>% add_rowindex(), by = ".row")

set.seed(1)

nhl_rink_plot() +
  geom_point(
    data = rpart_pred %>% sample_n(1000),
    aes(x = coord_x, y = coord_y, col = prediction),
    alpha = .4,
    cex = 2
  ) +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "top") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x  = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y  = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank()
  )


# ------------------------------------------------------------------------------

get_curve_data <- function(x, label) {
  x %>%
    collect_predictions(parameters = select_best(x, metric = "roc_auc")) %>%
    roc_curve(on_goal, .pred_yes) %>%
    mutate(model = label)
}

bind_rows(
  get_curve_data(rpart_tune, "CART"),
  get_curve_data(glmnet_tune, "glmnet"),
  get_curve_data(ranger_tune, "Random Forest"),
) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_abline(alpha = .3) +
  geom_step() +
  lims(x = 0:1, y = 0:1) +
  coord_equal()

