# ------------------------------------------------------------------------------
# SCRIPT 9: SPARSE PLS (sPLS) COM 1ª DERIVADA E SNV
# ------------------------------------------------------------------------------

if(!require(spls)) install.packages("spls")
if(!require(caret)) install.packages("caret")
if(!require(prospectr)) install.packages("prospectr") # Necessário para os tratamentos

library(spls)
library(caret)
library(prospectr)

# 1. Carregar dados
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])

# Ajuste de compatibilidade: Forçar nomes sequenciais limpos nas colunas espectrais
colnames(X) <- paste0("V", 1:ncol(X))

# 2. PRÉ-TRATAMENTO DOS DADOS (SNV + 1ª Derivada)
X_snv <- standardNormalVariate(X)
X_tratado <- savitzkyGolay(X_snv, m = 1, p = 2, w = 11) # m=1 aplica a 1ª Derivada

# 3. Divisão Estratificada 70/30
set.seed(123)
indices_treino <- createDataPartition(Y, p = 0.70, list = FALSE)

X_treino <- X_tratado[indices_treino, ]
Y_treino <- Y[indices_treino]
X_teste  <- X_tratado[-indices_treino, ]
Y_teste  <- Y[-indices_treino]

# 4. Ajustar Modelo sPLS
# eta (0.1 a 0.9): controla a severidade da seleção de variáveis (sparsity)
# K: número de componentes ocultos/latentes
modelo_spls <- spls(X_treino, Y_treino, eta = 0.7, K = 4)

# 5. --- EXTRAIR VALORES PREDITOS (VALIDAÇÃO EXTERNA) ---
predicoes <- predict(modelo_spls, newx = X_teste)

# Garantir o formato de vetor numérico simples para os cálculos
valores_preditos <- as.vector(predicoes)
valores_reais <- Y_teste

# 6. Construir Tabela de Resultados
tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# 7. Exibir desempenho no console
print("--- TABELA DE PREDIÇÕES SPARSE PLS (sPLS) ---")
print(tabela_resultados)

# Calcular R² e RMSE globais para comparação
r2_tot <- cor(tabela_resultados$Valor_Real, tabela_resultados$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_resultados$Valor_Real - tabela_resultados$Valor_Predito)^2))

cat("\n====================================================")
cat("\nDesempenho sPLS (1ª Derivada) -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

# 8. Gráfico de Validação
plot(valores_reais, valores_preditos,
     xlab = "API Real (Laboratório)", ylab = "API Predita (sPLS)",
     main = "Sparse PLS com Pré-tratamento (1ª Derivada)",
     pch = 19, col = "darkblue")
abline(0, 1, col = "red", lwd = 2)

View(tabela_resultados)
