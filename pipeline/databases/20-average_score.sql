-- Creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN target_user INT
)
BEGIN
    DECLARE avg_result DECIMAL(5,2);

    SELECT IFNULL(AVG(score), 0)
    INTO avg_result
    FROM corrections
    WHERE user_id = target_user;

    UPDATE users
    SET average_score = avg_result
    WHERE id = target_user;
END;
//

DELIMITER ;