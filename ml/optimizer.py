import pandas as pd
import numpy as np
from ortools.sat.python import cp_model  # type: ignore
import os

class RoommateOptimizer:
    """
    A constraint programming optimizer for assigning students to university hall rooms
    using Google OR-Tools CP-SAT solver.
    """

    def __init__(self, hall_data_path, student_data_path):
        """
        Initialize the optimizer with data paths.

        Args:
            hall_data_path (str): Path to hall-room dataset CSV
            student_data_path (str): Path to student dataset CSV
        """
        self.hall_data_path = hall_data_path
        self.student_data_path = student_data_path
        self.hall_df = None
        self.student_df = None
        self.compatibility_matrix = None
        self.model = None
        self.solver = None
        self.solution = None
        self.student_room = None
        self.same_room = None

    def load_data(self):
        """
        Load hall and student datasets from CSV files.
        """
        print("Loading hall and student data...")

        # Load hall data
        self.hall_df = pd.read_csv(self.hall_data_path)
        print(f"Loaded {len(self.hall_df)} rooms from hall dataset")

        # Load student data
        self.student_df = pd.read_csv(self.student_data_path)
        print(f"Loaded {len(self.student_df)} students from student dataset")

        # Rename columns for consistency
        self.student_df = self.student_df.rename(columns={
            'Matric Number': 'Student_ID',
            'Current academic level': 'Level'
        })

        # Add Department column if not present (extract from Matric Number)
        if 'Department' not in self.student_df.columns:
            # Extract department from matric number (assuming format like 21AB028409)
            self.student_df['Department'] = self.student_df['Student_ID'].str[2:4]

    def calculate_compatibility(self, student1, student2):
        """
        Calculate compatibility score between two students based on their preferences.

        Args:
            student1 (pd.Series): First student data
            student2 (pd.Series): Second student data

        Returns:
            float: Compatibility score (0-1, higher is better)
        """
        if student1['Student_ID'] == student2['Student_ID']:
            return 0  # No self-compatibility

        # Define preference columns (excluding ID, Gender, Level, Department)
        preference_cols = [
            'Preferred living environment', 'Organize personal space frequency',
            'Productive environment', 'Atmosphere created', 'Invite friends frequency',
            'Comfortable with roommate bringing guests', 'Sleep lights preference',
            'Fan speed preference', 'Preferred study time', 'Okay with roommate studying late night'
        ]

        # Count matching preferences
        matches = 0
        total_prefs = 0

        for col in preference_cols:
            if col in student1.index and col in student2.index:
                if pd.notna(student1[col]) and pd.notna(student2[col]):
                    if student1[col] == student2[col]:
                        matches += 1
                    total_prefs += 1

        # Return normalized score
        return matches / total_prefs if total_prefs > 0 else 0

    def build_compatibility_matrix(self):
        """
        Build a compatibility matrix for all student pairs.
        """
        print("Building compatibility matrix...")

        n_students = len(self.student_df)
        self.compatibility_matrix = np.zeros((n_students, n_students))

        # Pre-extract preference columns for efficiency
        preference_cols = [
            'Preferred living environment', 'Organize personal space frequency',
            'Productive environment', 'Atmosphere created', 'Invite friends frequency',
            'Comfortable with roommate bringing guests', 'Sleep lights preference',
            'Fan speed preference', 'Preferred study time', 'Okay with roommate studying late night'
        ]

        # Convert to numpy array for faster operations
        pref_data = self.student_df[preference_cols].values

        for i in range(n_students):
            for j in range(i+1, n_students):
                # Count matching preferences
                matches = 0
                total_prefs = 0

                for k in range(len(preference_cols)):
                    val_i = pref_data[i, k]
                    val_j = pref_data[j, k]
                    if pd.notna(val_i) and pd.notna(val_j):
                        if val_i == val_j:
                            matches += 1
                        total_prefs += 1

                # Return normalized score
                score = matches / total_prefs if total_prefs > 0 else 0
                self.compatibility_matrix[i, j] = score
                self.compatibility_matrix[j, i] = score

        print(f"Compatibility matrix built ({n_students}x{n_students})")

    def setup_model(self):
        """
        Set up the CP-SAT model with variables and constraints.
        """
        print("Setting up CP-SAT model...")

        self.model = cp_model.CpModel()
        n_students = len(self.student_df)
        n_rooms = len(self.hall_df)

        # Create assignment variables: student_room[i][j] = 1 if student i assigned to room j
        self.student_room = {}
        for i in range(n_students):
            for j in range(n_rooms):
                self.student_room[(i, j)] = self.model.NewBoolVar(f'student_{i}_room_{j}')

        # Create variables for pairs being in the same room (only for compatible pairs)
        self.same_room = {}
        compatibility_threshold = 0.5  # Only consider pairs with compatibility > 0.5
        for i in range(n_students):
            for j in range(i+1, n_students):
                if self.compatibility_matrix[i, j] > compatibility_threshold:
                    self.same_room[(i, j)] = self.model.NewBoolVar(f'same_room_{i}_{j}')

        # Constraint 1: Each student assigned to exactly one room
        for i in range(n_students):
            self.model.Add(sum(self.student_room[(i, j)] for j in range(n_rooms)) == 1)

        # Constraint 2: Room capacity not exceeded
        for j in range(n_rooms):
            room_capacity = self.hall_df.iloc[j]['Rooms_Per_Floor']  # Assuming this is capacity
            self.model.Add(sum(self.student_room[(i, j)] for i in range(n_students)) <= room_capacity)

        # Constraint 3: Gender matching - students only assigned to halls matching their gender
        for i in range(n_students):
            student_gender = self.student_df.iloc[i]['Gender']
            for j in range(n_rooms):
                hall_gender = self.hall_df.iloc[j]['Gender']
                if student_gender != hall_gender:
                    self.model.Add(self.student_room[(i, j)] == 0)

        # Constraint 4: Define same_room variables (only for high compatibility pairs)
        # same_room[i][j] = 1 iff student i and j are in the same room
        for (i, j) in self.same_room.keys():
            # same_room[i][j] <= min over k of (assignment[i][k] + assignment[j][k] - 1)
            for k in range(n_rooms):
                # If both i and j are in room k, then same_room must be 1
                self.model.Add(self.student_room[(i, k)] + self.student_room[(j, k)] - 1 <= self.same_room[(i, j)])
            # If same_room is 1, then there exists some k where both are in room k
            # This is automatically satisfied

        # Objective: Maximize total compatibility between compatible roommates
        objective_terms = []
        for (i, j), var in self.same_room.items():
            compat_score = int(self.compatibility_matrix[i, j] * 100)  # Scale for integer
            objective_terms.append(compat_score * var)

        if objective_terms:
            self.model.Maximize(sum(objective_terms))
        else:
            # If no compatible pairs, just find any feasible solution
            self.model.Maximize(sum(self.student_room.values()))

    def solve(self):
        """
        Solve the optimization model.

        Returns:
            bool: True if solution found, False otherwise
        """
        print("Solving optimization model...")

        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 300  # 5 minute timeout

        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found! Status: {status}")
            print(f"Objective value: {self.solver.ObjectiveValue()}")
            return True
        else:
            print(f"No solution found. Status: {status}")
            return False

    def extract_solution(self):
        """
        Extract the assignment solution into a DataFrame.

        Returns:
            pd.DataFrame: Student room allocation DataFrame
        """
        print("Extracting solution...")

        allocations = []

        for i in range(len(self.student_df)):
            for j in range(len(self.hall_df)):
                if self.solver.Value(self.student_room[(i, j)]) == 1:
                    student = self.student_df.iloc[i]
                    room = self.hall_df.iloc[j]

                    allocation = {
                        'Student_ID': student['Student_ID'],
                        'Gender': student['Gender'],
                        'Department': student.get('Department', ''),
                        'Level': student['Level'],
                        'Hall_Name': room['Hall_Name'],
                        'Wing': room['Wing'],
                        'Floor': room['Floor'],
                        'Room_Number': room['Room_Number'],
                        'Room_Type': room['Room_Type']
                    }
                    allocations.append(allocation)
                    break

        return pd.DataFrame(allocations)

    def save_allocation(self, output_path):
        """
        Save the allocation to CSV file.

        Args:
            output_path (str): Path to save the allocation CSV
        """
        allocation_df = self.extract_solution()
        allocation_df.to_csv(output_path, index=False)
        print(f"Allocation saved to {output_path}")
        print(f"Total students allocated: {len(allocation_df)}")

        # Print summary statistics
        hall_summary = allocation_df.groupby(['Hall_Name', 'Room_Number']).size().reset_index(name='Occupancy')
        print("\nRoom occupancy summary:")
        print(hall_summary.describe())

        return allocation_df

    def run_optimization(self, output_path='../data/Student_Room_Allocation.csv'):
        """
        Run the complete optimization process.

        Args:
            output_path (str): Path to save the results

        Returns:
            pd.DataFrame: Allocation results
        """
        # Load data
        self.load_data()

        # Build compatibility matrix
        self.build_compatibility_matrix()

        # Set up model
        self.setup_model()

        # Solve
        if self.solve():
            # Save results
            return self.save_allocation(output_path)
        else:
            print("Optimization failed - no feasible solution found")
            return None


if __name__ == "__main__":
    # Initialize optimizer
    optimizer = RoommateOptimizer(
        hall_data_path='../data/Hall_Room_Dataset.csv',
        student_data_path='../data/roommate_dataset_final.csv'
    )

    # Run optimization
    result = optimizer.run_optimization()

    if result is not None:
        print("Optimization completed successfully!")
    else:
        print("Optimization failed.")
