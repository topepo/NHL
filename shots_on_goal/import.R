
library(tidyverse)
library(tidymodels)
library(lubridate)
library(conflicted)

tidymodels_prefer()
conflict_prefer("lag", "dplyr")

theme_set(theme_bw())

# ------------------------------------------------------------------------------

df_raw <-
  readRDS("Data/plays_2017.rds") %>%
  select(event,
         eventCode,
         eventTypeId,
         description,
         secondaryType,
         gameWinningGoal,
         emptyNet,
         penaltySeverity,
         penaltyMinutes,
         eventIdx,
         eventId,
         period,
         periodType,
         ordinalNum,
         periodTime,
         periodTimeRemaining,
         dateTime,
         x,
         y,
         id,
         name,
         link,
         triCode,
         row.Names,
         Assist1,
         Assist2,
         Blocker,
         DrewBy,
         Goalie,
         Hittee,
         Hitter,
         Loser,
         PenaltyOn,
         PlayerID,
         Scorer,
         Shooter,
         Winner,
         gamePk,
         gameType,
         away_team,
         home_team,
         strength.code,
         strength.name,
         goals.away,
         goals.home,
         ServedBy)


###############################
# Filter and Pre-Process Data #
###############################

# Goal Line at x = 89 feet
# Goal Width 3 ft (use 2.9999 ft to avoid NaN arctan result)
# For shots from behind net (x > 89) assume shot from goal line x = 89


# Pre-process Data
# Remove Period = 5 and time = 00:00 data as those are shootout events
shots_on_goal <-
  df_raw %>%
  filter(grepl("(SHOT)|(^GOAL$)", eventTypeId),
         !is.na(x),
         !is.na(y),
         !(gameType %in% c("PR", "A")),
         !(period == 5 & periodTime == '00:00')) %>%
  mutate(
    shot_time = as.integer(substr(periodTime, 1, 2)) * 60 + as.integer(substr(periodTime, 4, 5)),
    shot_distance = sqrt(y ^ 2 + (89 - abs(x)) ^ 2),
    shot_angle = case_when(abs(x) <= 89 ~ atan((abs(y) - 2.999) / (89 - abs(x))) * 180 / pi,
                           TRUE ~ atan((abs(y) - 2.9999) / (0)) * 180 / pi),
    on_goal = case_when(eventTypeId %in% c("SHOT", "GOAL") ~ "yes", TRUE ~ "no"),
    on_goal = factor(on_goal, levels = c("yes", "no"))
    ) %>%
  # ----------------------------------------------------------------------------
  # Look at the teams shots on goal proportion before the shot being predicted.
  arrange(gamePk, triCode, dateTime) %>%
  group_by(gamePk, triCode) %>%
  mutate(
    prev_shot = dplyr::lag(on_goal),
    prev_shot = as.numeric(prev_shot == "yes"),
    prev_shot = if_else(is.na(prev_shot), 1/2, prev_shot),
    running_count = cumsum(prev_shot),
    running = running_count/row_number()
    ) %>%
  ungroup() %>%
  # ------------------------------------------------------------------------------
  select(
    on_goal,
    running,
    shot_type = secondaryType,
    shot_distance,
    shot_angle,
    goals_away = goals.away,
    goals_home = goals.home,
    team_home = home_team,
    game_type = gameType,
    shooter = Shooter,
    period_type = periodType,
    date_time = dateTime,
    gamePk,
    team = name,
    coord_x = x,
    coord_y = y
  ) %>%
  mutate(
    shooter = gsub("([[:space:]])|([[:punct:]])", "_", tolower(shooter)),
    shooter = gsub("__", "_", shooter),
    team_home = as.numeric(team == team_home),
    team = gsub(" ", "_", tolower(team)),
    period_type = tolower(period_type),
    regular_season = as.numeric(game_type == "R"),
    behind_goal_line = as.numeric(abs(coord_x) >= 89)
  ) %>%
  arrange(gamePk, date_time) %>%
  select(-shot_type, -date_time, -gamePk, -game_type) %>%
  mutate(across(where(is.character), as.factor)) %>%
  relocate(on_goal)

save(shots_on_goal, file = "shots_on_goal.RData", compress = "bzip2", compression_level = 9)

# ------------------------------------------------------------------------------

# https://thewincolumn.ca/2021/01/15/r-tutorial-creating-an-nhl-rink-using-the-tidyverse/
devtools::source_url("https://raw.githubusercontent.com/mrbilltran/the-win-column/master/nhl_rink_plot.R")

NHL_red <- NHL_blue <- NHL_light_blue <- rgb(0, 0, 0, .2)


set.seed(1)
nhl_rink_plot() +
  geom_point(
    data = shots_on_goal %>% sample_n(200),
    aes(x = coord_x, y = coord_y, col = on_goal),
    alpha = .4,
    cex = 2
    ) +
  scale_color_brewer(palette = "Dark2") +
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

