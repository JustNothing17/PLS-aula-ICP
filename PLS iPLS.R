# ------------------------------------------------------------------------------
# SCRIPT 3: iPLS DE INTERVALO ÚNICO (SEM TRATAMENTO)
# ------------------------------------------------------------------------------
if(!require(pls)) install.packages("pls")
if(!require(caret)) install.packages("caret")

library(pls)
library(caret)

# 1. Carregar dados
print("Selecione o arquivo de dados...")
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])
colnames(X) <- paste0("V", 1:ncol(X))

# 2. Divisão Estratificada 70/30 aplicada na matriz completa primeiro
df_completo <- data.frame(API = Y, Espectro = I(X))
set.seed(123)
indices_treino <- createDataPartition(df_completo$API, p = 0.70, list = FALSE)
dados_treino <- df_completo[indices_treino, ]
dados_teste  <- df_completo[-indices_treino, ] # Limpado caracteres ocultos de sintaxe

# 3. Executar a varredura iPLS nas fatias de dados de treino
tamanho_janela <- 100
num_intervalos <- floor(ncol(X) / tamanho_janela)
erros_intervalos <- numeric(num_intervalos)

for(i in 1:num_intervalos) {
  inicio <- ((i - 1) * tamanho_janela) + 1
  fim <- i * tamanho_janela
  X_treino_local <- dados_treino$Espectro[, inicio:fim]
  
  df_local <- data.frame(API = dados_treino$API, Espectro = I(X_treino_local))
  mod_local <- plsr(API ~ Espectro, data = df_local, ncomp = 3, validation = "CV", method = "simpls")
  erros_intervalos[i] <- min(RMSEP(mod_local)$val[2, , ])
}

# 4. Identificar o melhor intervalo matemático
melhor_int <- which.min(erros_intervalos)
cat("\n[INFO] O melhor intervalo espectral isolado foi o bloco:", melhor_int, "\n")

col_inicio <- ((melhor_int - 1) * tamanho_janela) + 1
col_fim <- melhor_int * tamanho_janela

# 5. Ajustar o modelo definitivo APENAS com as colunas do melhor intervalo
X_treino_otimo <- dados_treino$Espectro[, col_inicio:col_fim]
df_treino_otimo <- data.frame(API = dados_treino$API, Espectro = I(X_treino_otimo))

modelo_ipls_final <- plsr(API ~ Espectro, data = df_treino_otimo, ncomp = 3, scale = TRUE, method = "simpls")

# 6. --- EXTRAIR VALORES PREDITOS (VALIDAÇÃO EXTERNA) ---
X_teste_otimo <- dados_teste$Espectro[, col_inicio:col_fim]
df_teste_otimo <- data.frame(Espectro = I(X_teste_otimo))

# SOLUÇÃO DO ERRO: Forçar a saída do predict do pacote 'pls' a se tornar um vetor numérico plano
predicoes_brutas <- predict(modelo_ipls_final, ncomp = 3, newdata = df_teste_otimo)
valores_preditos <- as.vector(predicoes_brutas)
valores_reais <- dados_teste$API

# 7. Construir Tabela de Resultados
tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

print("--- TABELA DE PREDIÇÕES iPLS (INTERVALO ÚNICO) ---")
print(tabela_resultados)

# Calcular R² e RMSE para comparação (Agora sem erros de dimensões)
r2_tot <- cor(tabela_resultados$Valor_Real, tabela_resultados$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_resultados$Valor_Real - tabela_resultados$Valor_Predito)^2))

cat("\n====================================================")
cat("\nDesempenho iPLS Único -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

# 8. Gráfico de Validação
plot(valores_reais, valores_preditos,
     xlab = "API Real (Laboratório)", ylab = "API Predita iPLS",
     main = "Regressão PLS por iPLS (Melhor Intervalo)",
     pch = 19, col = "darkmagenta")
abline(0, 1, col = "red", lwd = 2)

View(tabela_resultados)
