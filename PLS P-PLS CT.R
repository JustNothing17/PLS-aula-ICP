# ------------------------------------------------------------------------------
# SCRIPT 2 CORRIGIDO: PENALIZED PLS (P-PLS) COM 1ª DERIVADA e SNV
# ------------------------------------------------------------------------------

# 1. Carregar pacotes necessários
if(!require(plsRglm)) install.packages("plsRglm")
if(!require(caret)) install.packages("caret")
if(!require(prospectr)) install.packages("prospectr")

library(plsRglm)
library(caret)
library(prospectr)

# 2. Carregar dados
print("Selecione o arquivo de dados...")
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])
colnames(X) <- paste0("V", 1:ncol(X))

# [ADICIONADO] Pré-tratamento Quimiométrico Necessário para as Pontas (<20 e >50)
X_snv <- standardNormalVariate(X)
X_tratado <- savitzkyGolay(X_snv, m = 1, p = 2, w = 11)

# 3. Divisão Estratificada 70/30 baseada em Y
set.seed(123)
indices_treino <- createDataPartition(Y, p = 0.70, list = FALSE)

X_treino <- X_tratado[indices_treino, ]
Y_treino <- Y[indices_treino]
X_teste  <- X_tratado[-indices_treino, ]
Y_teste  <- Y[-indices_treino]

# 4. Treinar o modelo P-PLS (nt = número máximo de componentes)
num_componentes <- 6
modelo_ppls <- plsRglm(dataY = Y_treino, dataX = X_treino, nt = num_componentes, modele = "pls-glm-gaussian")

# 5. EXTRAIR VALORES PREDITOS
df_X_teste <- as.data.frame(X_teste)
todas_predicoes <- predict(modelo_ppls, newdata = df_X_teste)

# SOLUÇÃO DO ERRO: Verifica se o resultado tem dimensões (matriz). 
# Se tiver, extrai a coluna correta. Se não tiver (vetor), usa o próprio objeto.
if (!is.null(dim(todas_predicoes))) {
  valores_preditos <- as.vector(todas_predicoes[, num_componentes])
} else {
  valores_preditos <- as.vector(todas_predicoes)
}

# O valor real correto vem do vetor Y de teste isolado anteriormente
valores_reais <- Y_teste

# 6. Tabela de Resultados
tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

print("--- PREDIÇÕES PENALIZED PLS (P-PLS CORRIGIDO) ---")
print(tabela_resultados)

# 7. Calcular R² e RMSE para comparação (CORRIGIDO)
r2_tot <- cor(tabela_resultados$Valor_Real, tabela_resultados$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_resultados$Valor_Real - tabela_resultados$Valor_Predito)^2))

cat("\n====================================================")
cat("\nDesempenho GERAL P-PLS -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

# Verificação focalizada nas extremidades críticas do seu projeto
print("--- VERIFICAÇÃO  ---")
print(tabela_resultados)

# 8. Gráfico de Validação
plot(valores_reais, valores_preditos,
     xlab = "API Real (Laboratório)", ylab = "API Predita (P-PLS)",
     main = "Regressão por Penalized PLS (P-PLS)",
     pch = 19, col = "darkmagenta")
abline(0, 1, col = "red", lwd = 2)

View(tabela_resultados)