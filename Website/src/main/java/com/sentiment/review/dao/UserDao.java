package com.sentiment.review.dao;

import com.sentiment.review.model.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;

/*
 * Data Access Object Class
 * Store and retrieve data from MySql database
 */
public class UserDao {
	
	// Create MySql Database connection
	private Connection getConnection() {
		Connection connection = null;
		
		try {
			Class.forName("com.mysql.cj.jdbc.Driver");
			connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/website", "root", "");
			
		} catch(Exception e) {
			e.printStackTrace();
		}
		return connection;
	}
	
	// Insert record in database
	public int submitForm(User user) throws ClassNotFoundException {
		String INSERT_USERS_SQL = "INSERT INTO reviews(first_name, last_name, email, stars, review)" +
				" VALUES(?, ?, ?, ?, ?);";
		
		int result = 0;		
		try {
			Connection connection = getConnection();
			// Create statement using Connection object
			PreparedStatement preparedStatement = connection.prepareStatement(INSERT_USERS_SQL);
			preparedStatement.setString( 1, user.getFirst_name());
			preparedStatement.setString( 2,  user.getLast_name());
			preparedStatement.setString( 3,  user.getEmail());
			preparedStatement.setInt( 4,  user.getStars());
			preparedStatement.setString( 5,  user.getReview());
			
			System.out.println(preparedStatement);
			
			// Execute the query and update query
			result = preparedStatement.executeUpdate();
			
			connection.close();
			preparedStatement.close();
		}
		catch (SQLException e) { 
			// Process SQL Exception
			e.printStackTrace();
		}
		return result;
	}
	
	// Get records from database
	public ArrayList<User> getReviews() {
		ArrayList<User> data = new ArrayList<>(10);
		String GET_REVIEWS_SQL = "SELECT * FROM reviews ORDER BY id DESC LIMIT 10;";
		ResultSet rs = null;
		
		try {
			Connection connection = getConnection();
			Statement st = connection.createStatement();
			rs = st.executeQuery(GET_REVIEWS_SQL);
			
			while (rs.next()) {
				User user = new User();
				String idString = rs.getString(1);
				int id = Integer.parseInt(idString);
				
				String first_name = rs.getString(2);
				String last_name = rs.getString(3);
				String email = rs.getString(4);
				
				String starString = rs.getString(5);
				int stars = Integer.parseInt(starString);
				
				String review = rs.getString(6);
				
				user.setId(id);
				user.setFirst_name(first_name);
				user.setLast_name(last_name);
				user.setEmail(email);
				user.setStars(stars);
				user.setReview(review);
				
				data.add(user);
			}
			rs.close();
			connection.close();
			st.close();
		} catch(Exception e) {
			e.printStackTrace();
		}
		return data;
	}
}
