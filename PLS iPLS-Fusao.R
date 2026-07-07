# ------------------------------------------------------------------------------
# SCRIPT 8: iPLS POR FUSÃO DE INTERVALOS
# ------------------------------------------------------------------------------
if(!require(pls)) install.packages("pls")
if(!require(caret)) install.packages("caret")
if(!require(prospectr)) install.packages("prospectr")

library(pls)
library(caret)
library(prospectr)

# 1. Carregar dados e alterar nomes da tabela por problema de compatibilidade
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])
colnames(X) <- paste0("V", 1:ncol(X))

# 2. PRÉ-TRATAMENTO DOS DADOS
X_snv <- standardNormalVariate(X)
X_tratado <- savitzkyGolay(X_snv, m = 1, p = 2, w = 11) # 1ª Derivada

# 3. Divisão Estratificada 70/30
set.seed(123)
indices_treino <- createDataPartition(Y, p = 0.70, list = FALSE)

X_treino <- X_tratado[indices_treino, ]
Y_treino <- Y[indices_treino]
X_teste  <- X_tratado[-indices_treino, ]
Y_teste  <- Y[-indices_treino]

# 4. VARREDURA IPLS PARA SELEÇÃO DOS MELHORES INTERVALOS
tamanho_janela <- 150  # Tamanho de cada bloco espectral
num_intervalos <- floor(ncol(X_treino) / tamanho_janela)
erros_intervalos <- numeric(num_intervalos)

for(i in 1:num_intervalos) {
  inicio <- ((i - 1) * tamanho_janela) + 1
  fim <- i * tamanho_janela
  X_bloco <- X_treino[, inicio:fim]

  df_local <- data.frame(API = Y_treino, Espectro = I(X_bloco))
  mod_local <- plsr(API ~ Espectro, data = df_local, ncomp = 4, validation = "CV", method = "simpls")
  erros_intervalos[i] <- min(RMSEP(mod_local)$val[2, , ])
}

# Em vez de pegar só 1, vamos pegar os 3 MELHORES intervalos espectrais
melhores_int <- order(erros_intervalos, decreasing = FALSE)[1:3]
cat("Os 3 melhores intervalos químicos encontrados foram os blocos:", melhores_int, "\n")

# Reconstruir as matrizes unindo esses 3 blocos (Fusão de Intervalos)
indices_colunas <- c()
for(int in melhores_int) {
  inicio <- ((int - 1) * tamanho_janela) + 1
  fim <- int * tamanho_janela
  indices_colunas <- c(indices_colunas, inicio:fim)
}

X_treino_fusionado <- X_treino[, indices_colunas]
X_teste_fusionado  <- X_teste[, indices_colunas]

# 5. Modelagem PLS Definitiva com os Intervalos Fusionados
df_treino_final <- data.frame(API = Y_treino, Espectro = I(X_treino_fusionado))
df_teste_final  <- data.frame(API = Y_teste,  Espectro = I(X_teste_fusionado))

modelo_ipls_fusionado <- plsr(API ~ Espectro,
                              data = df_treino_final,
                              ncomp = 9,
                              scale = TRUE,
                              validation = "CV",
                              method = "simpls")

# 6. --- EXTRAIR VALORES PREDITOS (VALIDAÇÃO EXTERNA - 30%) ---
num_vl <- 5 # Verifique se o gráfico sugere aumentar ou diminuir
predicoes_finais <- predict(modelo_ipls_fusionado, ncomp = num_vl, newdata = df_teste_final)

valores_preditos <- as.vector(predicoes_finais)
valores_reais <- Y_teste

# Tabela de Resultados
tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# 7. Exibir desempenho
print("--- iPLS POR FUSÃO DE INTERVALOS ---")
print(tabela_resultados)

# Calcular R² e RMSE para comparação
r2_tot <- cor(tabela_resultados$Valor_Real, tabela_resultados$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_resultados$Valor_Real - tabela_resultados$Valor_Predito)^2))
cat("\n====================================================")
cat("\nDesempenho iPLS Único -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

# 8. Gráfico de Validação
plot(valores_reais, valores_preditos,
     xlab = "API Real", ylab = "API Predita (iPLS Fusionado)",
     main = "Regressão PLS por Fusão de Intervalos",
     pch = 19, col = "darkmagenta")
abline(0, 1, col = "red", lwd = 2)

View(tabela_resultados)
