# ------------------------------------------------------------------------------
# SCRIPT 1: SIMPLS DIRETAMENTE (SEM TRATAMENTO)
# ------------------------------------------------------------------------------

# 1. Carregar pacotes necessários
if(!require(pls)) install.packages("pls")
if(!require(caret)) install.packages("caret")
library(pls)
library(caret)

# 2. Carregar de dados. Uso de file.choose() para evitar que o arquivo não seja encontrado
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])
df_completo <- data.frame(API = Y, Espectro = I(X))

# ==============================================================================
# GARANTIR OS 70% PARA CRIAÇÃO DO MODELO E 30% PARA PREDIÇÃO
# ==============================================================================

set.seed(123) # Define uma semente para que a divisão seja sempre a mesma ao rodar

# createDataPartition cria índices correspondentes a 70% dos dados de forma estratificada
indices_treino <- createDataPartition(df_completo$API, p = 0.70, list = FALSE)

# Divisão exata dos subconjuntos
dados_treino <- df_completo[indices_treino, ]  # 70% para construir o modelo
dados_teste  <- df_completo[-indices_treino, ] # 30% restantes para testar a predição

# 3. Treinar o modelo PLS apenas com os 70%
modelo_simpls <- plsr(API ~ Espectro,
                      data = dados_treino,
                      ncomp = 10,
                      scale = TRUE,
                      validation = "CV",
                      method = "simpls")

# ==============================================================================
# FAZER A PLS INFORMAR OS VALORES PREDITOS
# ==============================================================================

# Define o número de Componentes Latentes (VLs) a ser utilizado
num_componentes <- 7

# Pegar as predições para os 30% de dados que o modelo NUNCA viu:
predicoes_teste <- predict(modelo_simpls, ncomp = num_componentes, newdata = dados_teste)

# O R gera uma matriz tridimensional. Vamos converter para um vetor simples para visualizar:
valores_preditos <- as.vector(predicoes_teste)
valores_reais    <- dados_teste$API

# Tabela de Resultados
tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# 7. Exibir desempenho
print(tabela_resultados)

# Calcular R² e RMSE para comparação
r2_tot <- cor(tabela_resultados$Valor_Real, tabela_resultados$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_resultados$Valor_Real - tabela_resultados$Valor_Predito)^2))

cat("\n====================================================")
cat("\nDesempenho sPLS -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

# 8. Gráfico de Validação
plot(valores_reais, valores_preditos,
     xlab = "API Real", ylab = "API Predita SIMPLS",
     main = "Regressão PLS por SIMPLS sem tratamento",
     pch = 19, col = "green")
abline(0, 1, col = "darkblue", lwd = 2)

plot(RMSEP(modelo_simpls))
View(tabela_resultados)

