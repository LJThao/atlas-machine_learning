-- Creates a trigger that decreases the quantity of an item after adding a new order.
DROP TRIGGER IF EXISTS trg_update_inventory;

DELIMITER $$

CREATE TRIGGER trg_update_inventory
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;
END;
$$

DELIMITER ;