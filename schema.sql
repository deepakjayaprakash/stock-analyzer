CREATE DATABASE `stock_db` /*!40100 DEFAULT CHARACTER SET latin1 */

CREATE TABLE `companies` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `symbol` varchar(40) DEFAULT NULL,
  `quandl_code` varchar(20) NOT NULL,
  `from_date` datetime DEFAULT NULL,
  `to_date` datetime DEFAULT NULL,
  `last_updated_at` datetime DEFAULT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `security_code` varchar(20) DEFAULT NULL,
  `industry` varchar(50) DEFAULT NULL,
  `actual_name` varchar(100) DEFAULT NULL,
  `isin` varchar(30) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id` (`id`),
  UNIQUE KEY `name` (`name`),
  UNIQUE KEY `quandl_code` (`quandl_code`),
  UNIQUE KEY `symbol` (`symbol`),
  KEY `quandl_code_index` (`quandl_code`)
) ENGINE=InnoDB AUTO_INCREMENT=9054 DEFAULT CHARSET=latin1


CREATE TABLE `time_series` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `company_id` bigint(20) default NULL,
  `symbol` varchar(40) DEFAULT NULL,
  `quandl_code` varchar(20) NOT NULL,
  `open` double default NULL,
  `close` double default NULL,
  `num_trades` double default NULL,
  `num_shares` double default null,
  `close_open_spread` double default NULL,
  `percentage_change` double default NULL,

  `trade_date` datetime DEFAULT NULL,
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  KEY `fk_company_id_time_series` (`company_id`),
  CONSTRAINT `fk_company_id_time_series` FOREIGN KEY (`company_id`) REFERENCES `companies` (`id`)
);