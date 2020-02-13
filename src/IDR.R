library(tidyverse)

colnames_boley <-c('chr', 'start', 'stop', 'name', 'score',
                   'strand', 'signalValue', 'pValue', 'qValue',
                   'peak', 'lidr', 'idr',
                   'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                   'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak'
                   )
colnames_midr <- c('chr', 'start', 'stop', 'name',
                   'score', 'strand', 'signalValue', 'pValue',
                   'qValue', 'peak', 'midr')

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

system("midr -m data/boley_merge.NarrowPeak -f data/boley_r1.NarrowPeak data/boley_r2.NarrowPeak -o results_archimedean")
system("midr -m data/boley_merge.NarrowPeak -f data/boley_r1.NarrowPeak data/boley_r2.NarrowPeak -o results_gaussian -mt gaussian")

samic_data <- read_tsv("results_archimedean/idr_boley_r1.NarrowPeak",
                       col_names = colnames_midr) %>%
  bind_cols(read_tsv("results_archimedean/idr_boley_r2.NarrowPeak",
                       col_names = colnames_midr) %>% 
              select(signalValue) %>%
              rename(r2_signalValue = signalValue)
  ) %>% 
  mutate(l2fc = log2(signalValue / r2_signalValue),
         rank_r1 = order(signalValue),
         rank_r2 = order(r2_signalValue)) %>% 
  select(-r2_signalValue)

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
  select(c( colnames_midr[1:length(colnames_midr) - 1], lidr, l2fc, rank_r1, rank_r2)) %>%
  rename(midr = lidr) %>% 
  mutate(method = "boley") %>% 
  bind_rows(samic_data %>%
              mutate(method = "samic")) %>% 
  bind_rows(idr_data %>%
              mutate(method = "gaussian")) %>% 
  mutate(rank = order(signalValue)) %>% 
  ggplot() +
  geom_point(aes(x = l2fc, y = midr, color = rank)) +
  facet_wrap(~method) +
  theme_bw()
ggsave("boley_vs_gaussian_vs_samic.pdf")

# boley
read_tsv("data/boleyidr2",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = rank_r1, y = rank_r2, color = 10^(-lidr)))

read_tsv("results/idr_boleyidr2",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = rank_r1, y = rank_r2, color = 10^(-lidr)))

read_tsv("results/idr_boleyidr2_t_0.0001_0.99",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = rank_r1, y = rank_r2, color = 10^(-lidr)))

read_tsv("results/idr_boleyidr2",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = rank_r1, y = rank_r2, color = midr))


read_tsv("results/idr_boleyidr2_0.99",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = 10^(-lidr), y = midr, color = rank_r1)) +
  theme_bw()

read_tsv("results/idr_boleyidr2_t_0.0001_0.99",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = 10^(-lidr), y = midr, color = rank_r1)) +
  theme_bw()

read_tsv("results/idr_boleyidr2_0.999",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = 10^(-lidr), y = midr, color = rank_r1)) +
  theme_bw()

read_tsv("results/idr_boleyidr2_0.9999",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  ggplot(data = .) +
  geom_point(aes(x = 10^(-lidr), y = midr, color = rank_r1)) +
  theme_bw()


read_tsv("results/idr_boleyidr2_0.99",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  mutate(l2FC = log2(r1_signalValue/r2_signalValue)) %>%
  mutate(lidr = 10^(-lidr)) %>%
  gather(lidr, midr, key = "method", value = "idr") %>%
  ggplot(data = .) +
  geom_point(aes(x = l2FC, y = idr, color = rank_r1)) +
  facet_wrap(~method) +
  theme_bw()

read_tsv("results/idr_boleyidr2_t_0.0001_0.99",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  arrange(chr, start, stop) %>%
  View()

read_tsv("results/idr_boleyidr2_t_0.0001_0.99",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  mutate(l2FC = log2(r1_signalValue/r2_signalValue)) %>%
  mutate(lidr = 10^(-lidr)) %>%
  gather(lidr, midr, key = "method", value = "idr") %>%
  ggplot(data = .) +
  geom_point(aes(x = l2FC, y = idr, color = rank_r1)) +
  facet_wrap(~method) +
  theme_bw()

read_tsv("results/idr_boleyidr2_0.999",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  mutate(l2FC = log2(r1_signalValue/r2_signalValue)) %>%
  mutate(lidr = 10^(-lidr)) %>%
  gather(lidr, midr, key = "method", value = "idr") %>%
  ggplot(data = .) +
  geom_point(aes(x = l2FC, y = idr, color = rank_r1)) +
  facet_wrap(~method) +
  theme_bw()

read_tsv("results/idr_boleyidr2_0.9999",
         col_names = c('chr', 'start', 'stop', 'name', 'score',
                       'strand', 'signalValue', 'pValue', 'qValue',
                       'peak', 'lidr', 'idr',
                       'r1_start', 'r1_stop', 'r1_signalValue', 'r1_peak',
                       'r2_start', 'r2_stop', 'r2_signalValue', 'r2_peak',
                       'midr')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  mutate(l2FC = log2(r1_signalValue/r2_signalValue)) %>%
  mutate(lidr = 10^(-lidr)) %>%
  gather(lidr, midr, key = "method", value = "idr") %>%
  ggplot(data = .) +
  geom_point(aes(x = l2FC, y = idr, color = rank_r1)) +
  facet_wrap(~method) +
  theme_bw()
  
colnames_midr <- c('chr', 'start', 'stop', 'strand', 'signalValue', 'peak', 'midr')

read_tsv("results_samic/idr_c1_r1.narrowPeak",
              col_names = colnames_midr) %>%
          mutate(replicate = 'r1') %>% 
  bind_rows( read_tsv("results_samic/idr_c1_r2.narrowPeak",
              col_names = colnames_midr) %>%
          mutate(replicate = 'r2')
          ) %>% 
  spread(key = replicate, value = signalValue) %>%
  mutate(rank_r1 = order(r1),
         rank_r2 = order(r2)) %>%
  mutate(l2FC = log2(r1/r2)) %>%
  left_join(read_tsv("data/boleyidr2",
         col_names = colnames_boley) %>%
  mutate(replicate = 'boley')) %>% 
  mutate(lidr = 10^(-lidr)) %>%
  gather(lidr, midr, key = "method", value = "idr") %>%
  ggplot(data = .) +
  geom_point(aes(x = l2FC, y = idr, color = rank_r1)) +
  facet_wrap(~method) +
  theme_bw()

read_tsv("results/idr_c1_r2.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1')  %>%
  bind_rows(read_tsv("results/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2')) %>%
  select(chr, start, stop, signalValue, idr, replicate) %>%
  mutate(replicate = as.factor(replicate)) %>%
  spread(key = replicate, value = signalValue)  %>%
  mutate(rank_r1 = rank(r1),
         rank_r2 = rank(r2)) %>%
  ggplot(data = .) +
  geom_point(aes(x = rank_r1, y = rank_r2, color = idr)) +
  scale_x_log10() +
  scale_y_log10()

read_tsv("results/idr_c1_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1',
         id = row_number()) %>%
  drop_na() %>%
  tail()
read_tsv("results/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2',
                 id = row_number()) %>%
  drop_na() %>%
  tail()

read_tsv("results_2_size_500/idr_c1_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1',
         id = row_number())  %>% 
  bind_rows(read_tsv("results_2_size_500/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2',
                 id = row_number())) %>%
  select(chr, start, stop, signalValue, idr, replicate) %>%
  pivot_wider(names_from = replicate, values_from = signalValue)  %>%
  mutate(rank_r1 = rank(r1),
         rank_r2 = rank(r2),
         state = as.factor(ifelse(idr < 0.05, "rep", "irep"))) %>%
  ggplot(mapping = aes(x = r1, y = r2)) +
  geom_point(aes(color = idr), alpha = 0.5) +
  geom_density2d(aes(group = state), color = "red") +
  theme_classic() +
  coord_quickmap()

read_tsv("results_2_size_1000/idr_c1_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1',
         id = row_number())  %>% 
  bind_rows(read_tsv("results_2_size_1000/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2',
                 id = row_number())) %>%
  select(chr, start, stop, signalValue, idr, replicate) %>%
  pivot_wider(names_from = replicate, values_from = signalValue)  %>%
  mutate(rank_r1 = rank(r1),
         rank_r2 = rank(r2),
         state = as.factor(ifelse(idr < 0.05, "rep", "irep"))) %>%
  ggplot(mapping = aes(x = r1, y = r2)) +
  geom_point(aes(color = idr), alpha = 0.5) +
  geom_density2d(aes(group = state), color = "red") +
  theme_classic() +
  coord_quickmap()

read_tsv("results_2/idr_c1_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1',
         id = row_number())  %>% 
  bind_rows(read_tsv("results_2/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2',
                 id = row_number())) %>%
  select(chr, start, stop, signalValue, idr, replicate) %>%
  pivot_wider(names_from = replicate, values_from = signalValue)  %>%
  mutate(rank_r1 = rank(r1),
         rank_r2 = rank(r2),
         state = as.factor(ifelse(idr < 0.05, "rep", "irep"))) %>%
  ggplot(mapping = aes(x = r1, y = r2)) +
  geom_point(aes(color = idr), alpha = 0.5) +
  geom_density2d(aes(group = state), color = "red") +
  theme_classic() +
  coord_quickmap()
  labs(title)

read_tsv("results_3/idr_c1_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1',
         id = row_number())  %>% 
  bind_rows(read_tsv("results_3/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2',
                 id = row_number())) %>%
  bind_rows(read_tsv("results_3/idr_c2_r1.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r3',
                 id = row_number())) %>%
  select(chr, start, stop, signalValue, idr, replicate, id) %>% filter(chr == "chr1") %>%
  ggplot(aes(color = replicate)) +
    geom_point(aes(y = signalValue, x = id, color = idr)) +
    facet_wrap(~replicate, nrow = 3) +
    theme_bw()
  

data <- read_tsv("results_3/idr_c1_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1',
         id = row_number())  %>% 
  bind_rows(read_tsv("results_3/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2',
                 id = row_number())) %>%
  bind_rows(read_tsv("results_3/idr_c2_r1.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r3',
                 id = row_number())) %>%
  select(chr, start, stop, signalValue, idr, replicate) %>%
  pivot_wider(names_from = replicate, values_from = signalValue)  %>%
  mutate(rank_r1 = rank(r1),
         rank_r2 = rank(r2),
         rank_r3 = rank(r3),
         state = as.factor(ifelse(idr < 0.05, "rep", "irep")))
ggplot(data, mapping = aes(x = r1, y = r2)) +
  geom_point(aes(color = idr), alpha = 0.5) +
  geom_density2d(aes(group = state), color = "red") +
  theme_classic() +
  coord_quickmap()
ggplot(data, mapping = aes(x = r2, y = r3, color = idr)) +
  geom_point(aes(color = idr), alpha = 0.5) +
  geom_density2d(aes(group = state), color = "red") +
  theme_classic() +
  coord_quickmap()
ggplot(data, mapping = aes(x = r1, y = r3, color = idr)) +
  geom_point(aes(color = idr), alpha = 0.5) +
  geom_density2d(aes(group = state), color = "red") +
  theme_classic() +
  coord_quickmap()



read_tsv("results_3/idr_c2_r1.narrowPeak",
         col_names = c('chr', 'start', 'stop', 'name',
                       'score', 'strand', 'signalValue', 'pValue',
                       'qValue', 'peak', 'idr')) %>%
  mutate(replicate = 'r1')  %>%
  bind_rows(read_tsv("results_3/idr_c1_r2.narrowPeak",
              col_names = c('chr', 'start', 'stop', 'name',
                            'score', 'strand', 'signalValue', 'pValue',
                            'qValue', 'peak', 'idr')) %>%
          mutate(replicate = 'r2')) %>%
  select(start, stop, signalValue, idr, replicate) %>%
  mutate(replicate = as.factor(replicate)) %>%
  spread(key = replicate, value = signalValue)  %>%
  mutate(rank_r1 = rank(r1),
         rank_r2 = rank(r2)) %>%
  ggplot(data = .) +
  geom_point(aes(x = rank_r1, y = rank_r2, color = idr)) +
  scale_x_log10() +
  scale_y_log10()


boley <- read_tsv("data/boleyidr2",
                  col_names = c('chr', 'start', 'stop', 'name', 'score',
                                'strand', 'signalValue', 'pValue', 'qValue',
                                'peak', 'lidr', 'idr',
                                'r1_start', 'r1_stop', 'r1_signalValue',
                                'r1_peak',
                                'r2_start', 'r2_stop', 'r2_signalValue',
                                'r2_peak')) %>%
  mutate(rank_r1 = order(r1_signalValue),
         rank_r2 = order(r2_signalValue)) %>%
  select('chr', 'start', 'stop', 'strand') %>%
  mutate(coords = paste(chr, start, stop, strand, sep="_")) %>%
  pull(coords)
midr <- read_tsv("results/idr_c1_r1.narrowPeak",
                 col_names = c('index', 'chr', 'start', 'stop', 'name',
                               'score', 'strand', 'signalValue', 'pValue',
                               'qValue', 'peak')) %>%
  select('chr', 'start', 'stop', 'strand') %>%
  mutate(coords = paste(chr, start, stop, strand, sep="_")) %>%
  pull(coords)
midr %in% boley %>% table()
midr %>% length()
boley %>% length()
boley[!(boley %in% midr)]

read_tsv("data/boleyidr2",
                  col_names = c('chr', 'start', 'stop', 'name', 'score',
                                'strand', 'signalValue', 'pValue', 'qValue',
                                'peak', 'lidr', 'idr',
                                'r1_start', 'r1_stop', 'r1_signalValue',
                                'r1_peak',
                                'r2_start', 'r2_stop', 'r2_signalValue',
                                'r2_peak')) %>%
  filter(chr == 'chr6' & start == 117906676 & stop == 117908211) %>%
  select('chr', 'start', 'stop', 'peak', 'lidr', 'idr','r1_start', 'r1_stop', 'r1_signalValue',
                                'r1_peak','r2_start', 'r2_stop', 'r2_signalValue',
                                'r2_peak')

