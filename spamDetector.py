import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vec.pkl','rb'))

def main():
	st.title("Email Spam Identification Application")
	st.write("This is a Machine Learning application to identify emails as spam or ham.")
	st.subheader("Identification")
	user_input=st.text_area("Enter an email to identify" ,height=150)
	if st.button("Identify"):
		if user_input:
			data=[user_input]
			print(data)
			vec=cv.transform(data).toarray()
			result=model.predict(vec)
			if result[0]==0:
				st.success(" No, Not A Spam Email")
			else:
				st.error(" Yes, Spam Email")
		else:
			st.write("Please enter an email to identify.")
main()
