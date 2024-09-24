# Load necessary library
library(dplyr)

# Define helper functions to generate random data
generate_dates <- function(n, start_date, end_date) {
  seq.Date(as.Date(start_date), as.Date(end_date), by = "day") %>%
    sample(n, replace = TRUE)
}

generate_names_and_purposes <- function(n) {
  categories <- list(
    Bars = list(names = c("Bar_Altes_Fahnenlager", "Bar_Kaschemme", "Bar_Werkstatt", "Alte Utting", "Bahnwaehrter Thiel"), purpose = "Entertainment"),
    Grocery_Shops = list(names = c("Rewe", "Edeka", "Netto", "Lidl", "Aldi"), purpose = "Groceries"),
    Supermarkets = list(names = c("Kaufland", "Real"), purpose = "Groceries"),
    Online_Shops = list(names = c("Amazon", "Zalando", "Otto", "Ebay", "AboutYou", "MyToys", "Shop_Bonprix", "Zara"), purpose = "Shopping"),
    Banks = list(names = c("Deutsche_Bank", "Commerzbank", "Sparkasse", "ING_DiBa", "DKB"), purpose = "Banking"),
    People = list(names = c("Alice", "Bob", "Charlie", "Dave", "Elon Musik", "Jose", "Xi Jingping", "Angela Merkel", "Lennard Riede", "Simon Han"), purpose = "Personal"),
    Apps = list(names = c("Uber", "Deutsche_Bahn", "Lufthansa", "Flink", "Gorillas", "Vueling"), purpose = "Travel"),
    Drug_Stores = list(names = c("DM", "Rossmann"), purpose = "Healthcare"),
    Restaurants = list(names = c("Restaurant_Mario", "Restaurant_Peking", "Mario Bello", "Honest Greens", "Hotpot Xinjiang", "Vinitus", "Tonkatsu"), purpose = "Dining"),
    Fast_Food_Chains = list(names = c("McDonalds", "Burger_King", "KFC", "Subway"), purpose = "Dining"),
    Hotels = list(names = c("Trump Tower", "Hilton", "Ritz-Carlton", "Four Seasons", "Waldorf Astoria", "Hotel Mama"), purpose = "Travel")
  )
  
  selected_categories <- sample(names(categories), n, replace = TRUE)
  
  names <- sapply(selected_categories, function(cat) sample(categories[[cat]]$names, 1))
  purposes <- sapply(selected_categories, function(cat) categories[[cat]]$purpose)
  
  list(names = names, purposes = purposes)
}

generate_sales_type <- function(n) {
  sales_types <- c(rep("Outgoing", round(n * 0.75)), rep("Ingoing", round(n * 0.25)))
  sample(sales_types, n)
}

generate_unique_ibans <- function(entities) {
  unique_entities <- unique(entities)
  ibans <- sapply(unique_entities, function(entity) {
    random_number <- floor(runif(1, min = 1e18, max = 1e19))
    iban <- paste0("DE", random_number)
    return(iban)
  })
  names(ibans) <- unique_entities
  ibans
}

generate_amounts <- function(n, sales_type, purposes, payers) {
  amounts <- numeric(n)
  for (i in 1:n) {
    if (sales_type[i] == "Ingoing") {
      if (payers[i] == "Pension Fund") {
        amounts[i] <- runif(1, min = 2000, max = 3000)
      } else if (payers[i] == "Government") {
        amounts[i] <- runif(1, min = 1000, max = 2000)
      } else if (payers[i] == "Family") {
        amounts[i] <- runif(1, min = 500, max = 1500)
      } else {
        amounts[i] <- runif(1, min = 10, max = 500)
      }
    } else {
      if (purposes[i] == "Groceries") {
        amounts[i] <- runif(1, min = 5, max = 100)
      } else if (purposes[i] == "Entertainment") {
        amounts[i] <- runif(1, min = 5, max = 150)
      } else if (purposes[i] == "Shopping") {
        amounts[i] <- runif(1, min = 10, max = 200)
      } else if (purposes[i] == "Banking") {
        amounts[i] <- runif(1, min = 50, max = 2500)
      } else if (purposes[i] == "Personal") {
        amounts[i] <- runif(1, min = 5, max = 250)
      } else if (purposes[i] == "Travel") {
        amounts[i] <- runif(1, min = 25, max = 1000)
      } else if (purposes[i] == "Healthcare") {
        amounts[i] <- runif(1, min = 5, max = 50)
      } else if (purposes[i] == "Dining") {
        amounts[i] <- runif(1, min = 5, max = 180)
      } else {
        amounts[i] <- runif(1, min = 1, max = 5000)
      }
    }
  }
  amounts <- ifelse(sales_type == "Outgoing", -amounts, amounts)
  round(amounts, 2)
}

generate_payers_and_payees <- function(n, sales_type, names) {
  payers <- ifelse(sales_type == "Outgoing", "Simon Han", sample(c("Pension Fund", "Government", "Family", "Lennard Riede", "Arsenii Vesnovskyi", "Marta Peregrin"), n, replace = TRUE))
  payees <- ifelse(sales_type == "Outgoing", names, "Simon Han")
  list(payers = payers, payees = payees)
}

generate_pension_transactions <- function(start_date, end_date, amount = 2500, category = "Pension") {
  dates <- seq.Date(as.Date(start_date), as.Date(end_date), by = "month")
  n <- length(dates)
  payers <- rep("Pension Fund", n)
  payees <- rep("Simon Han", n)
  purposes <- rep(category, n)
  sales_type <- rep("Ingoing", n)
  amounts <- rep(amount, n)
  data.frame(Date = dates, Payer = payers, Payee = payees, Purpose = purposes, Sales_Type = sales_type, Amount = amounts)
}

generate_rent_transactions <- function(start_date, end_date, amount = -900) {
  dates <- seq.Date(as.Date(start_date), as.Date(end_date), by = "month")
  n <- length(dates)
  payers <- rep("Simon Han", n)
  payees <- rep("Landlord", n)
  purposes <- rep("Rent", n)
  sales_type <- rep("Outgoing", n)
  amounts <- rep(amount, n)
  data.frame(Date = dates, Payer = payers, Payee = payees, Purpose = purposes, Sales_Type = sales_type, Amount = amounts)
}

# Generate synthetic data for retiree
set.seed(123)
n <- 1000
start_date <- "2020-01-01"
end_date <- "2023-12-31"
starting_budget <- 5000

names_and_purposes <- generate_names_and_purposes(n)
dates <- generate_dates(n, start_date, end_date)
sales_type <- generate_sales_type(n)
payers_and_payees <- generate_payers_and_payees(n, sales_type, names_and_purposes$names)
amounts <- generate_amounts(n, sales_type, names_and_purposes$purposes, payers_and_payees$payers)

# Generate unique IBANs for each payer and payee, including Pension Fund and Landlord
unique_entities <- unique(c(payers_and_payees$payers, payers_and_payees$payees, "Pension Fund", "Landlord"))
ibans <- generate_unique_ibans(unique_entities)

# Map IBANs to payers and payees
ibans_for_payers <- ibans[payers_and_payees$payers]
ibans_for_payees <- ibans[payers_and_payees$payees]

# Create the synthetic data frame
synthetic_data_retiree <- data.frame(
  Date = dates,
  Payer = payers_and_payees$payers,
  Payer_IBAN = ibans_for_payers,
  Payee = payers_and_payees$payees,
  Payee_IBAN = ibans_for_payees,
  Purpose = names_and_purposes$purposes,
  Sales_Type = sales_type,
  Amount = amounts
)

# Generate pension transactions
pension_transactions <- generate_pension_transactions(start_date, end_date)
pension_transactions$Payer_IBAN <- ibans["Pension Fund"]
pension_transactions$Payee_IBAN <- ibans["Simon Han"]

# Generate rent transactions
rent_transactions <- generate_rent_transactions(start_date, end_date)
rent_transactions$Payer_IBAN <- ibans["Simon Han"]
rent_transactions$Payee_IBAN <- ibans["Landlord"]

# Append pension and rent transactions to the synthetic data
synthetic_data_retiree <- bind_rows(synthetic_data_retiree, pension_transactions, rent_transactions)

# Calculate final balance starting from the specified budget
final_balance <- starting_budget + sum(synthetic_data_retiree$Amount)
cat("Final Balance:", final_balance, "\n")

# Export the generated data to a CSV file
write.csv(synthetic_data_retiree, "synthetic_data_retiree.csv", row.names = FALSE)
