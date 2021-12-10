library(tidymodels)
library(doMC)
library(embed)
library(discrim)
library(rules)
library(baguette)

# ------------------------------------------------------------------------------

registerDoMC(cores = 20)
tidymodels_prefer()
theme_set(theme_bw())

# ------------------------------------------------------------------------------

load("shots_on_goal.RData")

# ------------------------------------------------------------------------------

set.seed(1)
val_split <- validation_split(shots_on_goal)
grid_ctrl <- control_grid(parallel_over = "everything", save_pred = TRUE, save_workflow = TRUE)
resamp_ctrl <- control_resamples(parallel_over = "everything", save_pred = TRUE, save_workflow = TRUE)

cls_metrics <- metric_set(roc_auc, pr_auc, mn_log_loss, accuracy, sensitivity,
                          specificity, recall, precision, mcc, j_index,
                          f_meas, kap, ppv, npv)

# ------------------------------------------------------------------------------

effects_encode_recipe <-
  recipe(formula = on_goal ~ ., data = shots_on_goal) %>%
  update_role(starts_with("coord"), new_role = "coordinates") %>%
  step_lencode_mixed(shooter, outcome = vars(on_goal))

dummies_recipe <-
  effects_encode_recipe%>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

normalized_recipe <-
  dummies_recipe %>%
  step_normalize(all_numeric_predictors())

# ------------------------------------------------------------------------------

glmnet_spec <-
  logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

glmnet_workflow <-
  workflow() %>%
  add_recipe(normalized_recipe) %>%
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
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------

lda_tune <-
  discrim_linear() %>%
  fit_resamples(effects_encode_recipe, resamples = val_split, control = grid_ctrl,
                metrics = cls_metrics)

# ------------------------------------------------------------------------------

parsnip:::mlp_num_weights(50, 20, 2)

mlp_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_mode("classification") %>%
  set_engine("nnet", MaxNWts = 1200)

mlp_workflow <-
  workflow() %>%
  add_recipe(normalized_recipe) %>%
  add_model(mlp_spec)

mlp_tune <-
  tune_grid(
    mlp_workflow,
    resamples = val_split,
    grid = 25,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------

fda_tune <-
  discrim_flexible(prod_degree = tune()) %>%
  tune_grid(effects_encode_recipe,
            resamples = val_split,
            grid = 2,
            control = grid_ctrl,
            metrics = cls_metrics)

# ------------------------------------------------------------------------------

nb_tune <-
  naive_Bayes() %>%
  fit_resamples(effects_encode_recipe,
                resamples = val_split,
                control = grid_ctrl,
                metrics = cls_metrics)


# ------------------------------------------------------------------------------

knn_spec <-
  nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>%
  set_mode("classification")

knn_workflow <-
  workflow() %>%
  add_recipe(normalized_recipe) %>%
  add_model(knn_spec)

set.seed(9264)
knn_tune <-
  tune_grid(
    knn_workflow,
    resamples = val_split,
    grid = 10,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------

svm_spec <-
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("classification")

svm_workflow <-
  workflow() %>%
  add_recipe(normalized_recipe) %>%
  add_model(svm_spec)

set.seed(9264)
svm_tune <-
  tune_grid(
    svm_workflow,
    resamples = val_split,
    grid = 10,
    control = grid_ctrl,
    metrics = cls_metrics
  )


# ------------------------------------------------------------------------------

gam_spec <-
  gen_additive_mod() %>%
  set_mode("classification")

gam_workflow <-
  workflow() %>%
  add_recipe(effects_encode_recipe) %>%
  add_model(gam_spec, formula = on_goal ~ s(running) + s(shot_distance) + s(shot_angle))

gam_tune <-
  fit_resamples(
    gam_workflow,
    resamples = val_split,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# gam_fit <-
#   gam_workflow %>%
#   fit(data = analysis(val_split$splits[[1]]))
#
# gam_grid_recipe <-
#   recipe(formula = on_goal ~ ., data = shots_on_goal) %>%
#   step_profile(-on_goal, -shot_angle, profile = vars(shot_angle), grid = list(pctl = TRUE, len = 500)) %>%
#   prep() %>%
#   bake(new_data = NULL)
#
# predict(gam_fit, gam_grid_recipe, type = "prob") %>%
#   bind_cols(predict(gam_fit, gam_grid_recipe, type = "conf_int")) %>%
#   bind_cols(gam_grid_recipe) %>%
#   ggplot(aes(x = shot_angle, y = .pred_yes)) +
#   geom_path() +
#   geom_ribbon(aes(ymin = .pred_lower_yes, ymax = .pred_upper_yes), alpha = .1) +
#   ylim(0:1)


# ------------------------------------------------------------------------------

set.seed(9264)
cart_tune <-
  decision_tree(cost_complexity = tune(), min_n = tune()) %>%
  set_mode("classification") %>%
  tune_grid(effects_encode_recipe,
            resamples = val_split,
            grid = 20,
            control = grid_ctrl,
            metrics = cls_metrics)

set.seed(9264)
c5_tune <-
  decision_tree() %>%
  set_mode("classification") %>%
  set_engine("C5.0") %>%
  fit_resamples(effects_encode_recipe,
                resamples = val_split,
                control = grid_ctrl,
                metrics = cls_metrics)

# ------------------------------------------------------------------------------

bag_cart_spec <-
  bag_tree() %>%
  set_mode("classification") %>%
  set_engine("rpart", times = 25)

bag_cart_workflow <-
  workflow() %>%
  add_recipe(effects_encode_recipe) %>%
  add_model(bag_cart_spec)

bag_cart_tune <-
  fit_resamples(
    bag_cart_workflow,
    resamples = val_split,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------

bag_c5_spec <-
  bag_tree() %>%
  set_mode("classification") %>%
  set_engine("C5.0", times = 25)

bag_c5_workflow <-
  workflow() %>%
  add_recipe(effects_encode_recipe) %>%
  add_model(bag_c5_spec)

bag_c5_tune <-
  fit_resamples(
    bag_c5_workflow,
    resamples = val_split,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------

ranger_spec <-
  rand_forest(mtry = tune(),
              min_n = tune(),
              trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

ranger_workflow <-
  workflow() %>%
  add_recipe(effects_encode_recipe) %>%
  add_model(ranger_spec)

set.seed(9264)
ranger_tune <-
  tune_grid(
    ranger_workflow,
    resamples = val_split,
    grid = 30,
    control = grid_ctrl,
    metrics = cls_metrics
  )


# ------------------------------------------------------------------------------

xgboost_spec <-
  boost_tree(
    mtry = tune(),
    trees = 250,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    stop_iter = tune()
  ) %>%
  set_mode("classification")

xgboost_workflow <-
  workflow() %>%
  add_recipe(dummies_recipe) %>%
  add_model(xgboost_spec)

xgboost_param <-
  xgboost_workflow %>%
  parameters() %>%
  update(learn_rate = learn_rate(c(-6, -1)))

set.seed(9264)
xgboost_tune <-
  tune_grid(
    xgboost_workflow,
    resamples = val_split,
    grid = 30,
    param_info = xgboost_param,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------

c5_boost_spec <-
  boost_tree(trees = tune(), min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("C5.0")

c5_boost_workflow <-
  workflow() %>%
  add_recipe(effects_encode_recipe) %>%
  add_model(c5_boost_spec)

set.seed(9264)
c5_boost_tune <-
  tune_grid(
    c5_boost_workflow,
    resamples = val_split,
    grid = 30,
    control = grid_ctrl,
    metrics = cls_metrics
  )

# ------------------------------------------------------------------------------


rule_fit_spec <-
  rule_fit(
    mtry = tune(),
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    penalty = tune()
  ) %>%
  set_mode("classification")

rule_fit_workflow <-
  workflow() %>%
  add_recipe(dummies_recipe) %>%
  add_model(rule_fit_spec)

set.seed(9264)
rule_fit_tune <-
  tune_grid(
    rule_fit_workflow,
    resamples = val_split,
    grid = 30,
    control = grid_ctrl,
    metrics = cls_metrics
  )


# ------------------------------------------------------------------------------
#
# show_best(ranger_tune)
# show_best(rpart_tune)
# show_best(glmnet_tune)
#
# # ------------------------------------------------------------------------------
#
# # https://thewincolumn.ca/2021/01/15/r-tutorial-creating-an-nhl-rink-using-the-tidyverse/
# devtools::source_url("https://raw.githubusercontent.com/mrbilltran/the-win-column/master/nhl_rink_plot.R")
#
# NHL_red <- NHL_blue <- NHL_light_blue <- rgb(0, 0, 0, .2)
#
# # ------------------------------------------------------------------------------
#
# rpart_pred <-
#   rpart_tune %>%
#   collect_predictions(parameters = select_best(rpart_tune, metric = "roc_auc")) %>%
#   mutate(
#     prediction = ifelse(.pred_class == on_goal, "correct", "error"),
#     prediction = factor(prediction, levels = c("error", "correct"))
#     ) %>%
#   select(-on_goal) %>%
#   inner_join(shots_on_goal %>% add_rowindex(), by = ".row")
#
# set.seed(1)
#
# nhl_rink_plot() +
#   geom_point(
#     data = rpart_pred %>% sample_n(1000),
#     aes(x = coord_x, y = coord_y, col = prediction),
#     alpha = .4,
#     cex = 2
#   ) +
#   scale_color_brewer(palette = "Set1") +
#   theme(legend.position = "top") +
#   theme(
#     axis.title.x = element_blank(),
#     axis.text.x  = element_blank(),
#     axis.ticks.x = element_blank(),
#     axis.title.y = element_blank(),
#     axis.text.y  = element_blank(),
#     axis.ticks.y = element_blank(),
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank(),
#     panel.border = element_blank()
#   )
#
# # ------------------------------------------------------------------------------
#
# get_curve_data <- function(x, label) {
#   x %>%
#     collect_predictions(parameters = select_best(x, metric = "roc_auc")) %>%
#     roc_curve(on_goal, .pred_yes) %>%
#     mutate(model = label)
# }
#
# bind_rows(
#   get_curve_data(c5_tune, "C5.0"),
#   get_curve_data(cart_tune, "CART"),
#   get_curve_data(bag_c5_tune, "bagged C5.0"),
#   get_curve_data(bag_cart_tune, "bagged CART"),
#   get_curve_data(c5_boost_tune, "boost C5.0"),
#   get_curve_data(xgboost_tune, "xgboost"),
#   get_curve_data(rule_fit_tune, "rile fit"),
# ) %>%
#   ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
#   geom_abline(alpha = .3) +
#   geom_step() +
#   lims(x = 0:1, y = 0:1) +
#   coord_equal()
#
#
mods <- as_workflow_set(
  bag_c5 = bag_c5_tune,
  bag_cart = bag_cart_tune,
  c5_boost = c5_boost_tune,
  c5 = c5_tune,
  cart = cart_tune,
  fda = fda_tune,
  gam = gam_tune,
  glmnet = glmnet_tune,
  knn = knn_tune,
  lda = lda_tune,
  nb = nb_tune,
  ranger = ranger_tune,
  rule_fit = rule_fit_tune,
  svm = svm_tune,
  xgboost = xgboost_tune
)
