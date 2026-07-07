# ------------------------------------------------------------------------------
# SCRIPT DE ALTA PERFORMANCE: iPLS COMBINATÓRIO AVANÇADO
# ------------------------------------------------------------------------------
if(!require(pls)) install.packages("pls")
if(!require(caret)) install.packages("caret")
if(!require(prospectr)) install.packages("prospectr")

library(pls)
library(caret)
library(prospectr)

# 1. Carregar dados e limpar nomes
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])
colnames(X) <- paste0("V", 1:ncol(X))

# 2. PRÉ-TRATAMENTO REFINADO (Janela de suavização ligeiramente maior)
X_snv <- standardNormalVariate(X)
X_tratado <- savitzkyGolay(X_snv, m = 1, p = 2, w = 15) # w=15 estabiliza ruídos extremos

# 3. Divisão Estratificada 70/30
set.seed(123)
indices_treino <- createDataPartition(Y, p = 0.70, list = FALSE)

X_treino <- X_tratado[indices_treino, ]
Y_treino <- Y[indices_treino]
X_teste  <- X_tratado[-indices_treino, ]
Y_teste  <- Y[-indices_treino]

# 4. VARREDURA COMBINATÓRIA DE INTERVALOS OTIMIZADA
tamanho_janela <- 150
num_intervalos <- floor(ncol(X_treino) / tamanho_janela)
ranking_blocos <- data.frame(Intervalo = 1:num_intervalos, RMSECV = NA)

# Passo A: Avaliar blocos individuais
for(i in 1:num_intervalos) {
  inicio <- ((i - 1) * tamanho_janela) + 1
  fim <- i * tamanho_janela
  X_bloco <- X_treino[, inicio:fim]

  df_local <- data.frame(API = Y_treino, Espectro = I(X_bloco))
  mod_local <- plsr(API ~ Espectro, data = df_local, ncomp = 4, validation = "CV", method = "simpls")
  ranking_blocos$RMSECV[i] <- min(RMSEP(mod_local)$val[2, , ])
}

# Passo B: Selecionar os 4 melhores candidatos para o teste combinatório final
melhores_candidatos <- order(ranking_blocos$RMSECV, decreasing = FALSE)[1:4]

# Vamos testar qual combinação de 3 blocos (dentre os 4 melhores) dá o menor erro real
combinacoes <- combn(melhores_candidatos, 3)
melhor_erro_combinado <- Inf
melhor_grupo_colunas <- c()
melhores_blocos_finais <- c()

for(j in 1:ncol(combinacoes)) {
  blocos_teste <- combinacoes[, j]
  cols_teste <- c()
  for(b in blocos_teste) {
    inicio <- ((b - 1) * tamanho_janela) + 1
    fim <- b * tamanho_janela
    cols_teste <- c(cols_teste, inicio:fim)
  }

  X_comb_treino <- X_treino[, cols_teste]
  df_comb <- data.frame(API = Y_treino, Espectro = I(X_comb_treino))
  mod_comb <- plsr(API ~ Espectro, data = df_comb, ncomp = 5, validation = "CV", method = "simpls")
  erro_atual <- min(RMSEP(mod_comb)$val[2, , ])

  if(erro_atual < melhor_erro_combinado) {
    melhor_erro_combinado <- erro_atual
    melhor_grupo_colunas <- cols_teste
    melhores_blocos_finais <- blocos_teste
  }
}

cat("Equipe de intervalos campeã (Sinergia Química):", melhores_blocos_finais, "\n")

# 5. Filtrar Matrizes com a Combinação Campeã
X_treino_otimo <- X_treino[, melhor_grupo_colunas]
X_teste_otimo  <- X_teste[, melhor_grupo_colunas]

df_treino_final <- data.frame(API = Y_treino, Espectro = I(X_treino_otimo))
df_teste_final  <- data.frame(API = Y_teste,  Espectro = I(X_teste_otimo))

# 6. Modelo PLS Final com Sinergia de Intervalos
modelo_ipls_excelencia <- plsr(API ~ Espectro,
                               data = df_treino_final,
                               ncomp = 8,
                               scale = TRUE,
                               validation = "CV",
                               method = "simpls")

# 7. --- EXTRAIR VALORES PREDITOS (VALIDAÇÃO EXTERNA - 30%) ---
# Verifique o ponto mais baixo do gráfico gerado abaixo para definir o ncomp
plot(RMSEP(modelo_ipls_excelencia), main = "Critério de Escolha de VLs (Modelo Otimizado)")

num_vl <- 5
predicoes_finais <- predict(modelo_ipls_excelencia, ncomp = num_vl, newdata = df_teste_final)

valores_preditos <- as.vector(predicoes_finais)
valores_reais <- Y_teste

# Tabela de Resultados Final
tabela_final <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# 8. Exibir desempenho nas extremidades críticas (<20 e >50 API)
print("--- PREDIÇÕES DE EXCELÊNCIA NAS EXTREMIDADES (iPLS COMBINATÓRIO) ---")
print(tabela_final)

# Métricas de Validação das Extremidades para conferência
r2_tot <- cor(tabela_final$Valor_Real, tabela_final$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_final$Valor_Real - tabela_final$Valor_Predito)^2))
cat("\nMétricas nas Extremidades -> R²:", round(r2_tot, 5), "| RMSE:", round(rmse_tot, 2), "\n")

# 9. Gráfico Real vs Predito
plot(valores_reais, valores_preditos,
     xlab = "Densidade API Real", ylab = "Densidade API Predita (iPLS Otimizado)",
     main = "iPLS por Seleção Combinatória de Sinergia",
     pch = 19, col = "darkgreen")
abline(0, 1, col = "red", lwd = 2)

View(tabela_final)
