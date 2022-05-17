package com.sentiment.review.controller;

import java.io.IOException;
import java.util.ArrayList;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.sentiment.review.dao.*;
import com.sentiment.review.model.*;

/**
 * Servlet implementation class UserServlet
 */
@WebServlet("/controller")
public class UserServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;
    
	private UserDao userDao = new UserDao();
    /**
     * @see HttpServlet#HttpServlet()
     */
    public UserServlet() {
        super();
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		System.out.println("UserServlet doGet");
		System.out.println(request.getContextPath());

		try {
			// Get data ArrayList
			ArrayList<User> data = userDao.getReviews();
			// Store data in request scope
			request.setAttribute("data", data);			
		} catch(Exception e) {
			e.printStackTrace();
		}
		
		// Forward to Reviews submission page
		RequestDispatcher dispatcher = request.getRequestDispatcher("/WEB-INF/views/submitForm.jsp");
		dispatcher.forward(request, response);
	}

	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		System.out.println("UserServlet doPost");
		String first_name = request.getParameter("first_name");
		String last_name = request.getParameter("last_name");
		String email = request.getParameter("email");
		String starsString = request.getParameter("stars");
		int stars = Integer.parseInt(starsString);
		String review = request.getParameter("review");
		
		User user = new User();
		user.setFirst_name(first_name);
		user.setLast_name(last_name);
		user.setEmail(email);
		user.setStars(stars);
		user.setReview(review);
		
		try {
			userDao.submitForm(user);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		RequestDispatcher dispatcher = request.getRequestDispatcher("/WEB-INF/views/submissionDetails.jsp");
		dispatcher.forward(request, response);
	}
	
	
	
	
	
	
	

}
