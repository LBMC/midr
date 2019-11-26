library(tidyverse)
library(mvtnorm)
library(gridExtra)
library(cowplot)

sim_data <- function(m = 1, cv = 0.5){
  sim_data  <- rbind(
    rmvnorm(
      n = 10000,
      mean = rep(0, 2),
      sigma = diag(2)
    ),
    rmvnorm(
      n = 10000,
      mean = rep(m, 2),
      sigma = matrix(c(1, cv, cv, 1), ncol = 2)
    )
  ) %>%
    data.frame() %>%
    (function(x) {
      names(x) <- c("z_1", "z_2")
      return(x)
    }) %>%
    as_tibble() %>%
    mutate(u_1 = pnorm(z_1),
           u_2 = pnorm(z_2),
           state = as.factor(c(rep("irep", 10000), rep("rep", 10000))))
  return(sim_data)
}


plot_data <- function(sim_data, m, cv) {
  empty <- ggplot() + geom_point(aes(1, 1), colour = "white") + theme(
    axis.ticks = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  )
  p <- ggplot(data = sim_data, aes(x = z_1, y = z_2)) +
    geom_density2d(
      alpha = 0.8,
      color = "gray",
      size = 3,
      h = 1
    ) +
    geom_density2d(aes(
      group = state,
      color = state,
      fill = state
    ),
    size = 1.5,
    h = 1) +
    coord_quickmap() +
    theme_bw()
  hist_top <- ggplot(data = sim_data, aes(x = z_1)) +
    geom_density() +
    geom_density(aes( group = state, color = state, fill = state )) +
    theme_classic() +
    coord_quickmap() +
    theme(legend.position = "none")
  hist_right <- ggplot(data = sim_data, aes(x = z_2)) +
    geom_density() +
    geom_density(aes(group = state, color = state, fill = state)) +
    coord_quickmap() +
    coord_flip() +
    labs(main = paste0("mean = ", m, ", cv = ", cv)) +
    theme_classic() +
  theme(legend.position = "none")
  grid.arrange(
    hist_top,
    empty,
    p,
    hist_right,
    ncol = 2,
    nrow = 2,
    widths = c(4, 1),
    heights = c(1, 4)
  )
}
sim_data(m = 2, cv = 0.8) %>%
plot_data(m = 2, cv = 0.8)

for (m in 1:2) {
  for (cv in seq(0, 1, by = 0.1)) {
    sim_data(m = m, cv = cv) %>%
      plot_data(m = m, cv = cv) %>%
      ggsave(
        filename = paste0("results/sim_mean_", m, "_cv_", cv, ".pdf"),
        plot = .,
        width = 12,
        height = 9
      )
  }
}

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


read_tsv("results/idr_c1_r1.narrowPeak",
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

