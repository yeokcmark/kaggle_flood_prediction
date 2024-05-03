rm(list=ls())
pacman::p_load(tidyverse,
               tidymodels,themis, skimr, usemodels, 
               broom, vip, DT,
               ranger, xgboost, glmnet, bonsai,
               doParallel)

## Load functions written by me ----
# function to define model parameters----
my_function_define_model_parameters <-
  function(my_workflowset, my_data)
  {
    result_df <- list()
    
    for (i in 1:length(my_workflowset$wflow_id))
    {
      param_name <- paste0("param_", my_workflowset$wflow_id[i])
      param <- 
        my_workflowset %>% 
        extract_workflow(id = my_workflowset$wflow_id[i]) %>% 
        extract_parameter_set_dials() %>% 
        finalize(my_data)
      result_df[[i]] <-list(param_name, param)
    }
    return(result_df)
  }

# function to insert model parameters into workflowset

my_function_insert_parameters <-
  function (my_workflowset, list_of_parameters)
  {
    for (i in 1:length(my_workflowset$wflow_id))
    {
      my_workflowset <-
        my_workflowset %>% 
        option_add(param_info = pluck(list_of_parameters[[i]][2],1),
                   id = my_workflowset$wflow_id[i])
    }
    
    return (my_workflowset)
    
  }

#### function to collect predictions
my_function_collect_predictions <-
  function(tune_workflowset, wflow_id_tbl)
  {
    pred_summary <- tibble()
    
    for (i in 1:nrow(wflow_id_tbl))
    {
      last_fit_pred <-
        tune_workflowset %>% 
        extract_workflow(id = as.character(wflow_id_tbl[i,1])) %>% 
        finalize_workflow(tune_workflowset %>%
                            extract_workflow_set_result(id = as.character(wflow_id_tbl[i,1])) %>% 
                            select_best(metric = "roc_auc")) %>% 
        last_fit(data_split) %>% 
        collect_predictions() %>% 
        dplyr::select(.pred_Yes,
                      turnover) %>% 
        mutate(algorithm = wflow_id_tbl[i,1])
      
      pred_summary<-bind_rows(pred_summary, last_fit_pred)
      
    }
    return(pred_summary)
  }

# import data
data_train <- read_csv("train.csv")
data_train$sum_all = rowSums(data_train[2:21])

data_test <-read_csv("test.csv")

data_fold <-
  data_train %>% 
  #slice_sample(prop = 0.001) %>% 
  vfold_cv(v =10)
skim(data_train)
glimpse(data_train)
# specify recipes

rec_base <-
  recipe(formula = FloodProbability ~.,
         data = data_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_zv(all_numeric_predictors())

rec_log <-
  rec_base %>% 
  step_log(all_numeric_predictors())

rec_yj <-
  rec_base %>% 
  step_YeoJohnson(all_numeric_predictors())

rec_norm <-
  rec_base %>% 
  step_normalize(all_numeric_predictors())

rec_pca <-
  rec_norm %>% 
  step_pca(all_numeric_predictors(), threshold = tune("pca_threshold"))

rec_poly <-
  rec_base %>% 
  step_poly(all_numeric_predictors(), degree = 2, role = "predictors") %>% 
  step_normalize(all_numeric_predictors())

rec_interact <-
  rec_base %>% 
  step_interact(terms = ~all_numeric_predictors():all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())
#rec_poly %>% prep() %>% juice()

## specify models

# random forest
spec_rf <-
  rand_forest() %>% 
  set_engine("ranger",
             importance = "permutation") %>% 
  set_mode("regression")

# xgboost
spec_xgb <-
  boost_tree(
    mtry = tune(), trees = tune(), min_n = tune(), tree_depth = tune(),
    learn_rate = tune(), loss_reduction = tune(), sample_size = tune(),
    stop_iter = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

## nnet
spec_nnet <- 
  mlp() %>% 
  set_engine("nnet") %>% 
  set_mode("regression")

# lightgbm
spec_lgb <-
  boost_tree() %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

# linear regression
spec_lm <-
  linear_reg()

#null
spec_null <-
  null_model() %>% 
  set_mode("regression") %>% 
  set_engine("parsnip")

# metrics

metric_model <-
  metric_set(rsq, rmse, mape)

# workflowset

base_set <- 
  workflow_set (
    # list pre-processor recipes
    list(interact=rec_interact,
         norm=rec_norm,
         yj=rec_yj,
         pca=rec_pca
    ),
    # list models under evaluation
    list(xgb=spec_xgb,
         null=spec_null
    ),
    cross = TRUE)

model_parameters <-
  my_function_define_model_parameters(base_set, data_train)

base_set <-
  my_function_insert_parameters(base_set, model_parameters)

set.seed(2024050301)
cl <- (detectCores()/2) - 1
cores <- cl*2

doParallel::registerDoParallel(cl, cores)

first_tune <-
  workflow_map(base_set,
               fn = "tune_grid",
               verbose = TRUE,
               seed = 2024050301,
               grid = 20,
               resamples = data_fold,
               metrics = metric_model,
               control = control_grid(verbose = TRUE,
                                      allow_par = TRUE,
                                      parallel_over = "everything"))
save(first_tune, file = "first_tune.Rda")

autoplot(first_tune, select_best = T, metric = "rsq")

first_tune %>% 
  workflowsets::rank_results(rank_metric = "rsq", select_best = T) %>% 
  filter(.metric == "rsq") %>% 
  dplyr::select(wflow_id, mean, std_err, rank) %>% 
  datatable() %>% 
  formatRound(columns = c("mean", "std_err"),
              digits = 5)


