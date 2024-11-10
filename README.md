# Streamlit Dashboard for Coherix Vision System BI Project

## **STAR-Situation**
![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/raw/main/Readme_image/8.png)

In the Cybertruck battery pack pilot line, I was responsible for the adhesive dispensing process in battery pack assembly. This process involved several steps:
1. A shuttle delivered the battery pack to the station.
2. A robot arm then dispensed adhesive along the perimeter and in five longitudinal lines on the pack (illustrated in the images).
3. Finally, after adhesive application, the robot arm positioned a top cover onto the pack.


#### **Goal**
The goal of this process was to ensure that each battery pack was securely sealed with adhesive to maintain product quality. To monitor this process, we used the Coherix Vision System, a tool designed for real-time inspection of adhesive and sealant beads. The system operates by projecting a laser onto the adhesive bead while an offset camera captures its profile, allowing us to measure the bead’s height and width accurately. As the process engineer, I frequently needed to provide daily insights into the consistency and effectiveness of this process, as requested by my manager.

## **STAR-Task**
![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/9.png)

To address my manager's daily questions and prove that the adhesive dispensing process was consistently effective, I applied my data analytics skills, using both Jupyter Notebook and Tableau for data analysis.
1. **Tableau Flow:** In Tableau, I created a Statistical Process Control (SPC) dashboard to track the adhesive dispensing process. By monitoring two key parameters—the total adhesive volume per pack and the pressure used to apply it—we ensured consistency in material usage and application. Tableau's live SQL connection allowed the dashboard to continuously update, providing real-time insights without manual intervention.
2. **Jupyter Notebook Flow:** Using Python in Jupyter Notebook, I calculated the daily First Pass Yield (FPY) and key process metrics like mean and standard deviation. In cases of failure, I would identify the affected pack's serial number, then perform manual checks through MOS and pack inspections using a Tangent PC and VNC viewer to determine the failure mode. However, this approach had limitations in efficiency and control, as it allowed for only single-pack inspections, which slowed down troubleshooting and process improvement.
#### **Challenges**
While Tableau's SPC dashboard updated automatically, the Jupyter Notebook workflow required daily manual runs and investigation of failure modes, which hindered efficiency and the scalability of our quality control.

### **STAR - Action**
![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/10.png)

1. **Transition to Streamlit**: Streamlit, which “turns data scripts into shareable web apps in minutes—all in pure Python, with no front-end experience required,” provided an ideal platform for automating my analysis and making it accessible. This allowed me to move away from the manual Jupyter Notebook workflows to a more efficient, real-time solution.

2. **Dashboard Deployment**: I converted my Jupyter Notebook Python code to a Streamlit app, creating an in-house dashboard to manage over 32,000 rows of SPC (Statistical Process Control) data. This dashboard aggregated data from multiple packs into a single chart, providing a holistic view of process trends and enhancing overall process control. It allowed for simultaneous analysis of RTV and 2K adhesive bead measurements, including height and width, which streamlined the analysis process. This solution enabled quick identification and analysis of failures, resulting in significant time savings and improving efficiency for process engineers by up to 90%.

#### **Dashboard Details**:

- **Original Powertrain Dashboard**:

![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/3.png)
- **New -SPC Dashboard**: The first Streamlit dashboard displayed aggregated SPC data in Tesla’s powertrain premises, providing comprehensive monitoring of bead measurements and allowing for rapid troubleshooting of failures.

![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/4.png)
![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/5.png)
- **New -FPY Dashboard**: The second dashboard focused on First Pass Yield (FPY) with additional tools like daily FPY tracking, pack zone heatmaps, and Pareto charts highlighting high-failure areas. It also included a date range filter, enabling engineers to analyze specific timeframes and assess the impact of process adjustments over time.

![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/6.png)
![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/7.png)

 **Organization-wide Accessibility and Continuous Updates**: Once deployed, the dashboard became accessible to the entire organization, providing live updates via SQL connections to the database. This eliminated the need for manual data pulls and allowed stakeholders to monitor process performance in real time, ensuring up-to-date insights across departments.

### **STAR - Result**

![Eight Image](https://github.com/Johnlee19990908/Streamlit_dashboard_TSLA/blob/main/Readme_image/11.png)

1. **Report Automation**: With the dashboard in place, my manager no longer needs to ask me daily about process performance; the dashboard provides real-time answers in 90% of scenarios, keeping the team informed effortlessly.

2. **Sustained Process Monitoring**: Even after my internship concludes, the dashboard will continue providing insights and tracking process metrics autonomously, ensuring continuous monitoring without manual intervention.

3. **Expansion to Other Stations**: My colleague Prasanna, who manages the wade seal dispense station and also uses Coherix for monitoring, has adapted my code to deploy a similar dashboard on his station. This expansion illustrates the dashboard’s versatility and applicability across other stations, enhancing operational efficiency across different processes.

4. **Scalability for Mass Production**: Automating processes with this dashboard sets the foundation for scalable automation across multiple machines on the manufacturing line. This translates to significant time savings for engineers in a high-volume production setting, improving operational efficiency across the entire assembly line.

5. **Yield Rate Improvement**: Using the dashboards, we identified improvement opportunities in zone 3, where Coherix sometimes scanned blank spots. By repositioning the imager further from the bead, we reduced these errors, potentially increasing the yield rate by 5-10%.


#### **Summary**  
Overall, the dashboard has proven invaluable in saving time, providing actionable insights, and uncovering process improvements that enhance performance, contributing to more efficient and effective operations in the adhesive dispensing process.

