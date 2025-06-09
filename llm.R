### API calls to LLMs ###

# library -----------------------------------------------------------------

library(tidyverse)
library(ellmer)
library(yardstick)

# data --------------------------------------------------------------------

# sample of paragraphs
uk_data <- arrow::read_parquet("uk_sample.parquet")

# prompts -----------------------------------------------------------------

prompt_system <- "You are a highly performant text classification machine. Your sole job is to classify judicial decisions. When presented with a text, you reply with the single most correct category from a closed set and nothing else. Before you choose the category, you think about the evidence for your choice."

prompt_user <- readLines("prompt.txt") |> paste0(collapse = "\n")

# classifiers -------------------------------------------------------------

# models
cl_gemini <- chat_google_gemini(
  model = "gemini-2.5-flash-preview-05-20",
  system_prompt = prompt_system,
  params = params(temperature = 0),
  echo = "none"
)

cl_gpt <- chat_openai(
  model = "gpt-4.1-2025-04-14",
  system_prompt = prompt_system,
  params = params(temperature = 0),
  echo = "none"
)

cl_ollama <- chat_ollama(
  model = "llama3.2",
  system_prompt = prompt_system,
  echo = "none"
)

# function to keep chat history clean
classify_text <- function(chat, text, prompt){
  chat <- chat$clone()$set_turns(list())
  chat$chat(interpolate(
    "{{prompt}}
    
    {{text}}
    
    "
  ))
}

# classify ----------------------------------------------------------------

out_gem <- list()
out_gpt <- list()
out_llama <- list()

for (i in 100:nrow(uk_data)){
  
  out_gem[[i]] <- classify_text(cl_gemini, uk_data$text[i], prompt_user)
  out_gpt[[i]] <- classify_text(cl_gpt, uk_data$text[i], prompt_user)
  out_llama[[i]] <- classify_text(cl_ollama, uk_data$text[i], prompt_user)
  
  Sys.sleep(5)
  
}

save(out_gem,out_gpt,out_llama, file = "results.RData")

# process -----------------------------------------------------------------

results <- tibble(
  gem = unlist(out_gem),
  gpt = unlist(out_gpt),
  llama = unlist(out_llama)
)

merged <- bind_cols(results, uk_data) |> 
  mutate(across(.cols = c(gem, gpt, llama), .fns = ~str_remove_all(., "\\<|\\>"))) |> 
  mutate(across(.cols = c(gem, gpt, llama, label), .fns = as.factor))

# evaluate ----------------------------------------------------------------

yardstick::accuracy(
  merged,
  truth = label,
  estimate = gem
)

yardstick::accuracy_vec(
  factor(unlist(out_gem)),
  factor(unlist(out_gpt))
)

yardstick::bal_accuracy_vec(
  factor(unlist(out_gem)),
  factor(unlist(out_gpt))
)

yardstick::mcc_vec(
  factor(unlist(out_gem)),
  factor(unlist(out_gpt))
)
