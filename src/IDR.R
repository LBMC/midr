library(tidyverse)

colnames_boley <-c('chr', 'start', 'stop', 'name', 'score',
                   'strand', 'signalValue', 'pValue', 'qValue',
                   'peak', 'lidr', 'idr',
                   'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                   'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak'
                   )
colnames_midr <- c('chr', 'start', 'stop', 'name',
                   'score', 'strand', 'signalValue', 'pValue',
                   'qValue', 'peak', 'midr', 'idrrr')

boley_data <- read_tsv("data/boleyidr2",
         col_names = colnames_boley)
boley_data %>% select(
    colnames_midr[1:(length(colnames_midr) - 1)]
  ) %>%
  write_delim(
    path = "data/boley_merge.NarrowPeak",
    delim = "\t",
    col_names = F
  )

boley_data %>% select(
    chr,
    r1_start,
    r1_stop,
    name,
    score,
    strand,
    r1_signalValue,
    pValue,
    qValue,
    r1_peak
  ) %>%
  rename(
    start = r1_start,
    stop = r1_stop,
    signalValue = r1_signalValue,
    peak = r1_peak
  ) %>%
  write_delim(
    path = "data/boley_r1.NarrowPeak",
    delim = "\t",
    col_names = F
  )
boley_data %>% select(
    chr,
    r2_start,
    r2_stop,
    name,
    score,
    strand,
    r2_signalValue,
    pValue,
    qValue,
    r2_peak
  ) %>%
  rename(
    start = r2_start,
    stop = r2_stop,
    signalValue = r2_signalValue,
    peak = r2_peak
  ) %>%
  write_delim(
    path = "data/boley_r2.NarrowPeak",
    delim = "\t",
    col_names = F
  )

system("midr -m data/boley_merge.NarrowPeak -f data/boley_r1.NarrowPeak data/boley_r2.NarrowPeak -mf max -o results_archimedean -v")
system("midr -m data/boley_merge.NarrowPeak -f data/boley_r1.NarrowPeak data/boley_r2.NarrowPeak -mf max -o results_gaussian -mt gaussian -v")

samic_data <- read_tsv("results_archimedean_max/idr_boley_r1.NarrowPeak",
                       col_names = colnames_midr) %>%
  bind_cols(read_tsv("results_archimedean_max/idr_boley_r2.NarrowPeak",
                       col_names = colnames_midr) %>% 
              select(signalValue) %>%
              rename(r2_signalValue = signalValue)
  ) %>% 
  mutate(l2fc = log2(signalValue / r2_signalValue),
         rank_r1 = order(signalValue),
         rank_r2 = order(r2_signalValue)) %>% 
  select(-r2_signalValue, -idrrr)

idr_data <- read_tsv("results_gaussian/idr_boley_r1.NarrowPeak",
                       col_names = colnames_midr) %>%
  bind_cols(read_tsv("results_gaussian/idr_boley_r2.NarrowPeak",
                       col_names = colnames_midr) %>% 
              select(signalValue) %>%
              rename(r2_signalValue = signalValue)
  ) %>% 
  mutate(l2fc = log2(signalValue / r2_signalValue),
         rank_r1 = order(signalValue),
         rank_r2 = order(r2_signalValue)) %>% 
  select(-r2_signalValue)

boley_data %>%
  mutate(l2fc = log2(r1_signalValue / r2_signalValue),
         lidr = 10^(-lidr),
         rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>% 
  select(c( colnames_midr[1:(length(colnames_midr) - 2)], lidr, l2fc, rank_r1, rank_r2)) %>%
  rename(midr = lidr) %>% 
  mutate(method = "boley") %>% 
  bind_rows(samic_data %>%
              mutate(method = "samic")) %>% 
  bind_rows(idr_data %>%
              mutate(method = "gaussian")) %>% 
  mutate(rank = order(signalValue)) %>% 
  ggplot() +
  geom_point(aes(x = l2fc, y = midr, color = log(signalValue))) +
  facet_wrap(~method) +
  theme_bw()
ggsave("boley_vs_gaussian_vs_samic_max.pdf")

colnames_midr <- c('chr', 'start', 'stop', 'name',
                   'score', 'strand', 'signalValue', 'pValue',
                   'qValue', 'peak', 'midr', 'idrrr')
list.files(
  path = "results/mold/2020_05_12_clone_data/R1_archimedian_size_1000/",
  pattern = "idr_clone.*",
  full.names = T) %>%  
  map(., read_tsv, col_names = colnames_midr) %>% 
  map(., select, signalValue) %>% 
  do.call(bind_cols, .) %>% 
  write_tsv("data/matrix_test.tsv", col_names = F)

