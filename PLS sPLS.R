# ------------------------------------------------------------------------------
# SCRIPT 4: SPARSE PLS (sPLS) SEM TRATAMENTO
# ------------------------------------------------------------------------------

if(!require(spls)) install.packages("spls")
if(!require(caret)) install.packages("caret")
library(spls)
library(caret)

# Carregar dados
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])

# Divisão Estratificada 70/30
set.seed(123)
indices_treino <- createDataPartition(Y, p = 0.70, list = FALSE)

X_treino <- X[indices_treino, ]; Y_treino <- Y[indices_treino]
X_teste  <- X[-indices_treino, ]; Y_teste  <- Y[-indices_treino]

# Ajustar Modelo sPLS
# O algoritmo precisa de parâmetros fixos ou previamente otimizados pelo cv.spls
modelo_spls <- spls(X_treino, Y_treino, eta = 0.5, K = 3)

# 5. --- EXTRAIR VALORES PREDITOS ---
predicoes <- predict(modelo_spls, newx = X_teste)

valores_preditos <- as.vector(predicoes)
valores_reais <- Y_teste

# 6. Construir Tabela de Resultados
tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# SOLUÇÃO DO ERRO: Remover amostras que falharam na predição antes de calcular as métricas
tabela_limpa <- na.omit(tabela_resultados)

# 7. Exibir desempenho no console
print("--- TABELA DE PREDIÇÕES SPARSE PLS ---")
print(tabela_limpa)

# Calcular R² e RMSE apenas com os dados válidos restantes

r2_tot <- cor(tabela_limpa$Valor_Real, tabela_limpa$Valor_Predito, use = "complete.obs")^2
rmse_tot <- sqrt(mean((tabela_limpa$Valor_Real - tabela_limpa$Valor_Predito)^2))

cat("\n====================================================")
cat("\nDesempenho sPLS -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

print(tabela_limpa)

# 8. Gráfico de Validação
plot(tabela_limpa$Valor_Real, tabela_limpa$Valor_Predito,
     xlab = "API Real ", ylab = "API Predito ",
     main = "Sparse PLS",
     pch = 19, col = "darkblue")
abline(0, 1, col = "red", lwd = 2)

View(tabela_limpa)
