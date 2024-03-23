import unittest
from src.rds import db


class TestDatabaseConnection(unittest.TestCase):
    def test_db_connection(self):
        """
        Test if the database is reachable by executing a simple SELECT statement.
        """
        try:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                self.assertIsNotNone(result)
                self.assertEqual(result[0], 1)
        except Exception as e:
            self.fail(f"Database connection failed with exception: {e}")


if __name__ == "__main__":
    unittest.main()
