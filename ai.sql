-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Sep 16, 2025 at 01:49 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `venus_ai`
--

-- --------------------------------------------------------

--
-- Stand-in structure for view `analytics_shipments`
-- (See below for the actual view)
--
CREATE TABLE `analytics_shipments` (
`date` date
,`year` int(4)
,`month` int(2)
,`therapy` varchar(65)
,`product_name` varchar(255)
,`skus` varchar(50)
,`dosage_form` varchar(55)
,`uom` varchar(255)
,`quantity` decimal(20,2)
,`value_inr` varchar(255)
,`price_inr_per_unit` double
,`indian_company` varchar(255)
,`supplier` varchar(255)
,`country` varchar(255)
,`city` varchar(255)
,`continent` varchar(100)
);

-- --------------------------------------------------------

--
-- Table structure for table `athena_messages`
--

CREATE TABLE `athena_messages` (
  `id` bigint(20) NOT NULL,
  `session_id` varchar(64) NOT NULL,
  `role` enum('user','assistant','system','tool') NOT NULL,
  `content` mediumtext NOT NULL,
  `tool_name` varchar(64) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `athena_sessions`
--

CREATE TABLE `athena_sessions` (
  `session_id` varchar(64) NOT NULL,
  `title` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `athena_summaries`
--

CREATE TABLE `athena_summaries` (
  `session_id` varchar(64) NOT NULL,
  `summary` text NOT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `export_data`
--

CREATE TABLE `export_data` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `date` date DEFAULT NULL,
  `therapy` varchar(65) DEFAULT NULL,
  `currency` varchar(10) DEFAULT NULL,
  `foreign_company` varchar(255) DEFAULT NULL,
  `continent` varchar(100) DEFAULT NULL,
  `foreign_country` varchar(255) DEFAULT NULL,
  `city` varchar(255) DEFAULT NULL,
  `foreign_port` varchar(255) DEFAULT NULL,
  `indian_company` varchar(255) DEFAULT NULL,
  `indian_port` varchar(255) DEFAULT NULL,
  `item_rate_inv` decimal(20,6) DEFAULT NULL,
  `mode_of_shipment` varchar(255) DEFAULT NULL,
  `sk_us` varchar(50) DEFAULT NULL,
  `total_amount_invoice_fc` decimal(20,2) DEFAULT NULL,
  `unit` varchar(255) DEFAULT NULL,
  `product_name_2` varchar(255) DEFAULT NULL,
  `quantity_2` decimal(20,2) DEFAULT NULL,
  `wa_rate` decimal(20,6) DEFAULT NULL,
  `product_with_sku` varchar(255) DEFAULT NULL,
  `sum_of_fob_inr` varchar(255) DEFAULT NULL,
  `type` varchar(55) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------

--
-- Structure for view `analytics_shipments`
--
DROP TABLE IF EXISTS `analytics_shipments`;

CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `analytics_shipments`  AS SELECT `r`.`date` AS `date`, year(`r`.`date`) AS `year`, month(`r`.`date`) AS `month`, `r`.`therapy` AS `therapy`, `r`.`product_name_2` AS `product_name`, `r`.`sk_us` AS `skus`, `r`.`type` AS `dosage_form`, `r`.`unit` AS `uom`, `r`.`quantity_2` AS `quantity`, `r`.`sum_of_fob_inr` AS `value_inr`, coalesce(`r`.`wa_rate`,`r`.`sum_of_fob_inr` / nullif(`r`.`quantity_2`,0)) AS `price_inr_per_unit`, `r`.`indian_company` AS `indian_company`, `r`.`foreign_company` AS `supplier`, `r`.`foreign_country` AS `country`, `r`.`city` AS `city`, `r`.`continent` AS `continent` FROM `export_data` AS `r` WHERE `r`.`quantity_2` > 0 AND `r`.`sum_of_fob_inr` > 0 ;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `athena_messages`
--
ALTER TABLE `athena_messages`
  ADD PRIMARY KEY (`id`),
  ADD KEY `ix_sess_time` (`session_id`,`created_at`);

--
-- Indexes for table `athena_sessions`
--
ALTER TABLE `athena_sessions`
  ADD PRIMARY KEY (`session_id`);

--
-- Indexes for table `athena_summaries`
--
ALTER TABLE `athena_summaries`
  ADD PRIMARY KEY (`session_id`);

--
-- Indexes for table `export_data`
--
ALTER TABLE `export_data`
  ADD PRIMARY KEY (`id`),
  ADD KEY `export_data_date_foreign_country_index` (`date`,`foreign_country`),
  ADD KEY `export_data_indian_company_foreign_company_index` (`indian_company`,`foreign_company`),
  ADD KEY `export_data_product_name_2_sk_us_index` (`product_name_2`,`sk_us`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `athena_messages`
--
ALTER TABLE `athena_messages`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `export_data`
--
ALTER TABLE `export_data`
  MODIFY `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `athena_messages`
--
ALTER TABLE `athena_messages`
  ADD CONSTRAINT `fk_msg_session` FOREIGN KEY (`session_id`) REFERENCES `athena_sessions` (`session_id`) ON DELETE CASCADE;

--
-- Constraints for table `athena_summaries`
--
ALTER TABLE `athena_summaries`
  ADD CONSTRAINT `fk_sum_session` FOREIGN KEY (`session_id`) REFERENCES `athena_sessions` (`session_id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
