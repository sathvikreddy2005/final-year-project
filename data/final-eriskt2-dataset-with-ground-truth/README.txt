There are a total of 909 users. The information of each user is saved in a ".json" different file. The "all_combined" folder includes all the user as ".jsons".

The "shuffled_ground_truth_labels.txt" contains the golden truth labels (0 for control and 1 for positive).

There are two categories of users: individuals suffering depression and control users. 
For each user, the collection contains a sequence of writings from that specific user along with the rest of the users that participated in the conversation (in chronological order). 
This approach allows systems to monitor ongoing interactions and make timely decisions based on the evolution of the conversation.

In the dataset, there are two types of instances: Submissions and Comments.

Submissions represent the primary posts created by users. These are the main content entries, often containing a title, a body, and additional metadata such as the author and date.

Comments are responses or replies made by users to a submission or to other comments, forming a hierarchical structure. Each comment includes information about the author, its content, and its parent (which could be another comment or a submission).

*Submission Fields:

-submissionId: A unique string identifier for the submission.
-author: The anonymous identifier of the user who created the submission.
-date: The timestamp indicating when the submission was created, in ISO 8601 format.
-body: The main content of the submission (the text body).
-title: The title summarizing the submission's content.
-number: The round number of the submission. A value of 0 indicates the first writing of the subject.
-targetSubject: The anonymous identifier of the target subject (the one to classify) related to the submission.
-comments: A list of comments associated with the submission, where each comment includes its own fields.

*Comment Fields:

-commentId: A unique string identifier for the comment, used for referencing.
-author: The anonymous identifier of the user who wrote the comment.
-date: The timestamp indicating when the comment was created, in ISO 8601 format.
-body: The text content of the comment.
-parent: The identifier of the parent item (either a submissionId or commentId) that the comment replies to.


For more information, check the eRisk website corresponding to this task (task 2): https://erisk.irlab.org/












