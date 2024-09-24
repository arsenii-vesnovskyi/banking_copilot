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
  sales_types <- c(rep("Outgoing", round(n * 0.86)), rep("Ingoing", round(n * 0.14)))
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
      if (payers[i] == "Parents") {
        amounts[i] <- runif(1, min = 500, max = 1500)
      } else if (payers[i] == "Scholarship Fund") {
        amounts[i] <- runif(1, min = 300, max = 1000)
      } else if (payers[i] == "Part-time Job") {
        amounts[i] <- runif(1, min = 300, max = 600)
      } else {
        amounts[i] <- runif(1, min = 10, max = 300)
      }
    } else {
      if (purposes[i] == "Groceries") {
        amounts[i] <- runif(1, min = 5, max = 50)
      } else if (purposes[i] == "Entertainment") {
        amounts[i] <- runif(1, min = 5, max = 75)
      } else if (purposes[i] == "Shopping") {
        amounts[i] <- runif(1, min = 10, max = 100)
      } else if (purposes[i] == "Banking") {
        amounts[i] <- runif(1, min = 25, max = 1000)
      } else if (purposes[i] == "Personal") {
        amounts[i] <- runif(1, min = 5, max = 125)
      } else if (purposes[i] == "Travel") {
        amounts[i] <- runif(1, min = 25, max = 500)
      } else if (purposes[i] == "Healthcare") {
        amounts[i] <- runif(1, min = 5, max = 25)
      } else if (purposes[i] == "Dining") {
        amounts[i] <- runif(1, min = 5, max = 90)
      } else {
        amounts[i] <- runif(1, min = 1, max = 2500)
      }
    }
  }
  amounts <- ifelse(sales_type == "Outgoing", -amounts, amounts)
  round(amounts, 2)
}

generate_payers_and_payees <- function(n, sales_type, names) {
  payers <- ifelse(sales_type == "Outgoing", "Marta Peregrin", sample(c("Parents", "Scholarship Fund", "Part-time Job", "Lennard Riede", "Arsenii Vesnovskyi"), n, replace = TRUE))
  payees <- ifelse(sales_type == "Outgoing", names, "Marta Peregrin")
  list(payers = payers, payees = payees)
}

generate_student_transactions <- function(start_date, end_date, amount = 800, category = "Allowance") {
  dates <- seq.Date(as.Date(start_date), as.Date(end_date), by = "month")
  n <- length(dates)
  payers <- rep("Parents", n)
  payees <- rep("Marta Peregrin", n)
  purposes <- rep(category, n)
  sales_type <- rep("Ingoing", n)
  amounts <- rep(amount, n)
  data.frame(Date = dates, Payer = payers, Payee = payees, Purpose = purposes, Sales_Type = sales_type, Amount = amounts)
}

generate_part_time_job_transactions <- function(start_date, end_date, amount = 500, category = "Salary") {
  dates <- seq.Date(as.Date(start_date), as.Date(end_date), by = "month")
  n <- length(dates)
  payers <- rep("Part-time Job", n)
  payees <- rep("Marta Peregrin", n)
  purposes <- rep(category, n)
  sales_type <- rep("Ingoing", n)
  amounts <- rep(amount, n)
  data.frame(Date = dates, Payer = payers, Payee = payees, Purpose = purposes, Sales_Type = sales_type, Amount = amounts)
}

generate_rent_transactions <- function(start_date, end_date, amount = 700) {
  dates <- seq.Date(as.Date(start_date), as.Date(end_date), by = "month")
  n <- length(dates)
  payers <- rep("Marta Peregrin", n)
  payees <- rep("Landlord", n)
  purposes <- rep("Rent", n)
  sales_type <- rep("Outgoing", n)
  amounts <- rep(-amount, n)
  data.frame(Date = dates, Payer = payers, Payee = payees, Purpose = purposes, Sales_Type = sales_type, Amount = amounts)
}

# Generate synthetic data for student
set.seed(123)
n <- 1000
start_date <- "2020-01-01"
end_date <- "2023-12-31"
starting_budget <- 2000

names_and_purposes <- generate_names_and_purposes(n)
dates <- generate_dates(n, start_date, end_date)
sales_type <- generate_sales_type(n)
payers_and_payees <- generate_payers_and_payees(n, sales_type, names_and_purposes$names)
amounts <- generate_amounts(n, sales_type, names_and_purposes$purposes, payers_and_payees$payers)

# Generate unique IBANs for each payer and payee, including Parents, Part-time Job, and Landlord
unique_entities <- unique(c(payers_and_payees$payers, payers_and_payees$payees, "Parents", "Part-time Job", "Landlord"))
ibans <- generate_unique_ibans(unique_entities)

# Map IBANs to payers and payees
ibans_for_payers <- ibans[payers_and_payees$payers]
ibans_for_payees <- ibans[payers_and_payees$payees]

# Create the synthetic data frame
synthetic_data_student <- data.frame(
  Date = dates,
  Payer = payers_and_payees$payers,
  Payer_IBAN = ibans_for_payers,
  Payee = payers_and_payees$payees,
  Payee_IBAN = ibans_for_payees,
  Purpose = names_and_purposes$purposes,
  Sales_Type = sales_type,
  Amount = amounts
)

# Generate allowance transactions
allowance_transactions <- generate_student_transactions(start_date, end_date)
allowance_transactions$Payer_IBAN <- ibans["Parents"]
allowance_transactions$Payee_IBAN <- ibans["Marta Peregrin"]

# Generate part-time job transactions
part_time_job_transactions <- generate_part_time_job_transactions(start_date, end_date)
part_time_job_transactions$Payer_IBAN <- ibans["Part-time Job"]
part_time_job_transactions$Payee_IBAN <- ibans["Marta Peregrin"]

# Generate rent transactions
rent_transactions <- generate_rent_transactions(start_date, end_date)
rent_transactions$Payer_IBAN <- ibans["Marta Peregrin"]
rent_transactions$Payee_IBAN <- ibans["Landlord"]

# Append allowance, part-time job, and rent transactions to the synthetic data
synthetic_data_student <- bind_rows(synthetic_data_student, allowance_transactions, part_time_job_transactions, rent_transactions)

# Calculate final balance starting from the specified budget
final_balance <- starting_budget + sum(synthetic_data_student$Amount)
cat("Final Balance:", final_balance, "\n")

# Export the generated data to a CSV file
write.csv(synthetic_data_student, "synthetic_data_student.csv", row.names = FALSE)