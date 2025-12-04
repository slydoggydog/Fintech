'use client';

import { useState, useEffect } from 'react';

interface BudgetItem {
  id: string;
  category: {
    id: string;
    name: string;
    type: string;
    icon: string;
  };
  monthlyAmount: number;
  weeklyAmount: number;
  spent: number;
  remaining: number;
}

export default function BudgetManager() {
  const [budgetItems, setBudgetItems] = useState<BudgetItem[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBudget();
  }, []);

  async function fetchBudget() {
    const response = await fetch('/api/budget');
    const data = await response.json();
    setBudgetItems(data);
    setLoading(false);
  }

  async function updateBudget(itemId: string, newAmount: number) {
    await fetch('/api/budget', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ itemId, monthlyAmount: newAmount }),
    });
    setEditingId(null);
    fetchBudget();
  }

  const groupedItems = {
    need: budgetItems.filter(item => item.category.type === 'need'),
    want: budgetItems.filter(item => item.category.type === 'want'),
    savings: budgetItems.filter(item => item.category.type === 'savings'),
  };

  const totals = {
    need: groupedItems.need.reduce((sum, item) => sum + item.monthlyAmount, 0),
    want: groupedItems.want.reduce((sum, item) => sum + item.monthlyAmount, 0),
    savings: groupedItems.savings.reduce((sum, item) => sum + item.monthlyAmount, 0),
  };

  const totalBudget = totals.need + totals.want + totals.savings;
  const monthlyIncome = 9000; // From their spreadsheet

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading...</div>;
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-8">Budget Settings</h1>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-blue-100 rounded-xl p-4">
          <p className="text-blue-800 text-sm">Monthly Income</p>
          <p className="text-2xl font-bold text-blue-900">${monthlyIncome.toLocaleString()}</p>
        </div>
        <div className="bg-red-100 rounded-xl p-4">
          <p className="text-red-800 text-sm">Needs (50% target)</p>
          <p className="text-2xl font-bold text-red-900">${totals.need.toLocaleString()}</p>
          <p className="text-xs text-red-700">{((totals.need / monthlyIncome) * 100).toFixed(1)}% of income</p>
        </div>
        <div className="bg-yellow-100 rounded-xl p-4">
          <p className="text-yellow-800 text-sm">Wants (30% target)</p>
          <p className="text-2xl font-bold text-yellow-900">${totals.want.toLocaleString()}</p>
          <p className="text-xs text-yellow-700">{((totals.want / monthlyIncome) * 100).toFixed(1)}% of income</p>
        </div>
        <div className="bg-green-100 rounded-xl p-4">
          <p className="text-green-800 text-sm">Savings (20% target)</p>
          <p className="text-2xl font-bold text-green-900">${totals.savings.toLocaleString()}</p>
          <p className="text-xs text-green-700">{((totals.savings / monthlyIncome) * 100).toFixed(1)}% of income</p>
        </div>
      </div>

      {/* Budget Categories */}
      {Object.entries(groupedItems).map(([type, items]) => (
        <div key={type} className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4 capitalize">
            {type === 'need' ? '🏠 Necessities' : type === 'want' ? '🎁 Wants' : '💰 Savings'}
          </h2>

          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xl">{item.category.icon}</span>
                  <span className="font-medium">{item.category.name}</span>
                </div>

                <div className="flex items-center gap-4">
                  {/* Spending Progress */}
                  <div className="w-32">
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          item.spent > item.monthlyAmount ? 'bg-red-500' : 'bg-green-500'
                        }`}
                        style={{
                          width: `${Math.min((item.spent / item.monthlyAmount) * 100, 100)}%`,
                        }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      ${item.spent.toFixed(0)} / ${item.monthlyAmount}
                    </p>
                  </div>

                  {/* Editable Amount */}
                  {editingId === item.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        className="w-24 px-2 py-1 border rounded"
                        autoFocus
                      />
                      <button
                        onClick={() => updateBudget(item.id, parseFloat(editValue))}
                        className="px-2 py-1 bg-green-500 text-white rounded text-sm"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => setEditingId(null)}
                        className="px-2 py-1 bg-gray-300 rounded text-sm"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => {
                        setEditingId(item.id);
                        setEditValue(item.monthlyAmount.toString());
                      }}
                      className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300"
                    >
                      ${item.monthlyAmount}/mo
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Remaining Income */}
      <div className="bg-gray-100 rounded-xl p-6">
        <div className="flex justify-between items-center">
          <span className="text-lg font-medium">Remaining After Budget</span>
          <span className={`text-2xl font-bold ${monthlyIncome - totalBudget >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            ${(monthlyIncome - totalBudget).toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
}
