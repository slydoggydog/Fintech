'use client';

import { useState, useEffect } from 'react';
import { format, startOfDay, endOfDay } from 'date-fns';

interface Transaction {
  id: string;
  name: string;
  merchantName?: string;
  amount: number;
  date: string;
  category?: {
    name: string;
    icon: string;
  };
  pending: boolean;
}

interface DailySpending {
  date: string;
  transactions: Transaction[];
  total: number;
  byCategory: { [key: string]: number };
}

export default function DailyDashboard() {
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [dailyData, setDailyData] = useState<DailySpending | null>(null);
  const [budgetComparison, setBudgetComparison] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDailyData();
  }, [selectedDate]);

  async function fetchDailyData() {
    setLoading(true);
    const dateStr = format(selectedDate, 'yyyy-MM-dd');
    const response = await fetch(`/api/spending/daily?date=${dateStr}`);
    const data = await response.json();
    setDailyData(data.daily);
    setBudgetComparison(data.budgetComparison);
    setLoading(false);
  }

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Date Navigation */}
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={() => setSelectedDate(d => {
            const newDate = new Date(d);
            newDate.setDate(newDate.getDate() - 1);
            return newDate;
          })}
          className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
        >
          ← Previous Day
        </button>

        <div className="text-center">
          <h1 className="text-2xl font-bold">
            {format(selectedDate, 'EEEE, MMMM d, yyyy')}
          </h1>
          <p className="text-gray-600">Daily Spending Review</p>
        </div>

        <button
          onClick={() => setSelectedDate(d => {
            const newDate = new Date(d);
            newDate.setDate(newDate.getDate() + 1);
            return newDate;
          })}
          className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300"
          disabled={selectedDate >= new Date()}
        >
          Next Day →
        </button>
      </div>

      {/* Daily Total Card */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
        <div className="flex justify-between items-center">
          <div>
            <p className="text-gray-600">Today&apos;s Spending</p>
            <p className="text-4xl font-bold text-red-600">
              ${dailyData?.total.toFixed(2) || '0.00'}
            </p>
          </div>
          <div className="text-right">
            <p className="text-gray-600">Daily Budget Target</p>
            <p className="text-2xl font-semibold text-green-600">
              ${budgetComparison?.dailyTarget.toFixed(2) || '0.00'}
            </p>
          </div>
        </div>

        {/* Budget Status Indicator */}
        <div className="mt-4">
          <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                (dailyData?.total || 0) > (budgetComparison?.dailyTarget || 0)
                  ? 'bg-red-500'
                  : 'bg-green-500'
              }`}
              style={{
                width: `${Math.min(
                  ((dailyData?.total || 0) / (budgetComparison?.dailyTarget || 1)) * 100,
                  100
                )}%`,
              }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-2">
            {(dailyData?.total || 0) <= (budgetComparison?.dailyTarget || 0)
              ? `✅ Under budget by $${((budgetComparison?.dailyTarget || 0) - (dailyData?.total || 0)).toFixed(2)}`
              : `⚠️ Over budget by $${((dailyData?.total || 0) - (budgetComparison?.dailyTarget || 0)).toFixed(2)}`}
          </p>
        </div>
      </div>

      {/* Transactions List */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Transactions</h2>

        {dailyData?.transactions.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No transactions for this day</p>
        ) : (
          <div className="space-y-3">
            {dailyData?.transactions.map((transaction) => (
              <div
                key={transaction.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">
                    {transaction.category?.icon || '❓'}
                  </span>
                  <div>
                    <p className="font-medium">
                      {transaction.merchantName || transaction.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {transaction.category?.name || 'Uncategorized'}
                      {transaction.pending && ' (Pending)'}
                    </p>
                  </div>
                </div>
                <p className={`font-bold ${transaction.amount > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  ${Math.abs(transaction.amount).toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Category Breakdown */}
      {dailyData && Object.keys(dailyData.byCategory).length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
          <h2 className="text-xl font-bold mb-4">Spending by Category</h2>
          <div className="space-y-2">
            {Object.entries(dailyData.byCategory).map(([category, amount]) => (
              <div key={category} className="flex justify-between items-center">
                <span>{category}</span>
                <span className="font-semibold">${(amount as number).toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
